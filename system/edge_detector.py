"""
边缘端：摄像头/视频流 + 跌倒检测 + MQTT 发布

职责（模拟部署在边缘设备上）：
  1. 加载训练好的 YOLOv5 模型（exp12/best.pt）
  2. 逐帧检测 normal / down
  3. 把检测结果通过 MQTT 发布到云端

用法（在项目根目录执行）：
  venv\\Scripts\\python.exe system\\edge_detector.py --source 你的视频.mp4
  venv\\Scripts\\python.exe system\\edge_detector.py --source 0                    # 摄像头
  venv\\Scripts\\python.exe system\\edge_detector.py --source 某图片.jpg           # 单张图
  venv\\Scripts\\python.exe system\\edge_detector.py --source dataset/images/val   # 图片目录
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                       # 让 `import config` 生效
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "yolov5"))  # 让 yolov5 的 models/utils 生效

import cv2
import numpy as np
import torch
import paho.mqtt.client as mqtt

import config

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def make_client():
    c = mqtt.Client(client_id=f"edge-{os.getpid()}", protocol=mqtt.MQTTv311)
    return c


def load_model(weights, device):
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    return model


def detect(frame, model, imgsz=640, conf_th=0.15):
    """对一帧做检测，返回 [{'class_id', 'class_name', 'conf', 'box'}, ...]"""
    img = letterbox(frame, imgsz, stride=model.stride, auto=model.pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(model.device).float() / 255.0
    img = img.unsqueeze(0)
    pred = model(img)
    det = non_max_suppression(pred, conf_th, 0.45, max_det=50)[0]

    names = model.names
    out = []
    if det is not None and len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in det.tolist():
            cid = int(cls)
            out.append({
                "class_id": cid,
                "class_name": names[cid],
                "conf": round(float(conf), 3),
                "box": [int(v) for v in xyxy],
            })
    return out


def draw_boxes(frame, dets):
    """在帧上画框（down=红色, normal=绿色），返回标注后的图像"""
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = (0, 0, 255) if d["class_name"] == "down" else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{d['class_name']} {d['conf']:.2f}",
                    (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


class StableDetector:
    """时域平滑器：减少"人物时有时无"的抖动，让识别更稳定。

    原理（滞回 + 保持）：
      - 只有置信度 >= enter_th 的检测才会被采纳，并切换当前类别；
      - 一旦采纳，即使后续若干帧没检测到，也保持住（hold_frames 帧），
        只有连续漏检超过 hold_frames 才降回 none。
      这样单帧的漏检/低置信度不会导致结果来回跳。
    """
    def __init__(self, enter_th=0.10, hold_frames=5):
        self.enter_th = enter_th
        self.hold_frames = hold_frames
        self.cls = "none"
        self.conf = 0.0
        self.miss = 0

    def update(self, dets):
        best = max(dets, key=lambda d: d["conf"]) if dets else None
        if best is not None and best["conf"] >= self.enter_th:
            self.cls = best["class_name"]
            self.conf = best["conf"]
            self.miss = 0
        else:
            self.miss += 1
            if self.miss > self.hold_frames:
                self.cls = "none"
                self.conf = 0.0
        return self.cls, self.conf


def main():
    ap = argparse.ArgumentParser(description="边缘端跌倒检测 + MQTT 发布")
    ap.add_argument("--source", default="0", help="视频文件 / 图片 / 目录 / 0(摄像头)")
    ap.add_argument("--weights", default=config.DEFAULT_WEIGHTS, help="模型权重路径")
    ap.add_argument("--device", default="0", help="0=GPU, cpu=CPU")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--conf", type=float, default=config.DOWN_CONF_THRESHOLD)
    ap.add_argument("--interval", type=float, default=0.5, help="发布节流：每帧之间最短间隔(秒)")
    ap.add_argument("--device-id", default="cam01", help="设备编号，写进消息")
    ap.add_argument("--broker", default=None, help="MQTT broker 地址（默认用 config 里的；云部署时填服务器公网 IP）")
    ap.add_argument("--no-show", action="store_true", help="不弹窗显示画面（仅发布 MQTT）")
    ap.add_argument("--save-frames", default=None, help="采集模式：把摄像头原始帧存到该目录（用于收集训练数据）")
    ap.add_argument("--save-interval", type=float, default=0.5, help="采集模式下存帧间隔(秒)")
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[边缘] 加载模型 {args.weights} ... device={dev}")
    model = load_model(args.weights, dev)

    # 连接 broker（broker 没启动也能继续跑，只是不发布）
    broker_host = args.broker or config.BROKER_HOST
    client = make_client()
    mqtt_ok = True
    try:
        client.connect(broker_host, config.BROKER_PORT, keepalive=60)
        client.loop_start()
        print(f"[边缘] 已连接 MQTT broker {broker_host}:{config.BROKER_PORT}，主题 {config.MQTT_TOPIC_EVENTS}")
    except Exception as e:
        mqtt_ok = False
        print(f"[边缘] [!] MQTT broker 连接失败（{e}），继续本地检测但不发布")

    # 打开输入源
    src = args.source
    if src.isdigit():
        # Windows 上用 DirectShow 后端打开摄像头，更稳更快
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        is_stream = True
    elif os.path.isdir(src):
        is_stream = False
        files = sorted([os.path.join(src, f) for f in os.listdir(src)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        cap = None
    else:
        cap = cv2.VideoCapture(src)
        is_stream = not src.lower().endswith((".jpg", ".jpeg", ".png"))

    last_pub = 0.0

    # 采集模式：把原始帧（不带框）存到目录，供后续标注/微调
    save_dir = args.save_frames
    save_counter = 0
    last_save = 0.0
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[边缘] 采集模式开启：原始帧保存到 {save_dir}，每 {args.save_interval}s 一张")

    def publish(stable_cls, stable_conf, dets):
        nonlocal last_pub
        now = time.time()
        if now - last_pub < args.interval:
            return
        last_pub = now
        payload = {
            "device": args.device_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "detections": dets,
            "class_name": stable_cls,
            "conf": stable_conf,
        }
        if mqtt_ok:
            client.publish(config.MQTT_TOPIC_EVENTS, json.dumps(payload))
        flag = f"{stable_cls}({stable_conf})" if stable_cls != "none" else "none"
        print(f"[边缘] 发布: {flag}")

    smoother = StableDetector(enter_th=args.conf, hold_frames=5)

    try:
        if is_stream:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                dets = detect(frame, model, args.img, args.conf)
                stable_cls, stable_conf = smoother.update(dets)
                publish(stable_cls, stable_conf, dets)
                if save_dir:  # 采集原始帧（画框之前，保证无标注）
                    now = time.time()
                    if now - last_save >= args.save_interval:
                        last_save = now
                        save_counter += 1
                        cv2.imwrite(os.path.join(save_dir, f"frame_{save_counter:06d}.jpg"), frame)
                if not args.no_show:
                    draw_boxes(frame, dets)
                    # 顶部状态条：显示稳定识别结果（不是逐帧抖动值）
                    color = (0, 0, 255) if stable_cls == "down" else (0, 255, 0)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (40, 40, 40), -1)
                    cv2.putText(frame, f"Status: {stable_cls}  {stable_conf:.2f}",
                                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow("Fall Detection (按 q 退出)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        else:
            # 图片文件或目录
            paths = files if cap is None else [src]
            for p in paths:
                frame = cv2.imread(p)
                if frame is None:
                    continue
                dets = detect(frame, model, args.img, args.conf)
                stable_cls, stable_conf = smoother.update(dets)
                publish(stable_cls, stable_conf, dets)
                time.sleep(args.interval)
            print("[边缘] 图片处理完成")
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        if is_stream:
            cap.release()
        cv2.destroyAllWindows()
    print("[边缘] 已退出")


if __name__ == "__main__":
    main()
