"""
模拟 IMU 边缘端：发布「原始加速度窗口」到 MQTT（云端推理用）。

与 imu_simulator.py（直接发 down/normal 结论）的区别：这里发的是原始 Ax/Ay/Az
采样序列（含重力），由云端提取特征 + 跑随机森林推理。用于演示「云端推理」。

用法（项目根目录）：
  # 持续跌倒
  venv\\Scripts\\python.exe system\\imu_raw_simulator.py --broker localhost --mode fall
  # 正常活动
  venv\\Scripts\\python.exe system\\imu_raw_simulator.py --broker localhost --mode normal
  # 每约 10 秒在「跌倒/正常」间切换（演示云端推理的实时响应）
  venv\\Scripts\\python.exe system\\imu_raw_simulator.py --broker localhost --mode alternate
"""
import argparse
import json
import time

import numpy as np
import paho.mqtt.client as mqtt

FS = 50          # 采样率 Hz
WINDOW_SEC = 2.0 # 每窗时长
N = int(FS * WINDOW_SEC)


def make_normal(rng):
    """站立行走：重力在 z，水平摆动为主（走路）。"""
    t = np.linspace(0, WINDOW_SEC, N)
    ax = 0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * rng.standard_normal(N)
    ay = 0.5 * np.sin(2 * np.pi * 0.8 * t) + 0.05 * rng.standard_normal(N)
    az = 1.0 + 0.15 * np.sin(2 * np.pi * 2.0 * t) + 0.05 * rng.standard_normal(N)
    return np.column_stack([ax, ay, az])


def make_fall(rng):
    """跌倒：站立 → 失重 → 垂直冲击 → 躺平（重力转到 x 轴）。"""
    w = np.zeros((N, 3))
    w[:10, 2] = 1.0 + 0.03 * rng.standard_normal(10)      # 站立
    w[10:25] = 0.05 * rng.standard_normal((15, 3))        # 失重（三轴≈0）
    w[25:32, 2] = -2.5 + 0.3 * rng.standard_normal(7)     # 垂直冲击
    w[32:, 0] = 1.0 + 0.04 * rng.standard_normal(N - 32)  # 躺平，重力落 x 轴
    w[32:, 1] = 0.04 * rng.standard_normal(N - 32)
    w[32:, 2] = 0.04 * rng.standard_normal(N - 32)
    return w


def make_window(mode, rng):
    if mode == "fall":
        return make_fall(rng)
    return make_normal(rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="fall/imu/raw")
    ap.add_argument("--mode", choices=["fall", "normal", "alternate"], default="fall")
    ap.add_argument("--device", default="imu01")
    ap.add_argument("--interval", type=float, default=2.5)
    args = ap.parse_args()

    c = mqtt.Client()
    c.connect(args.broker, args.port, keepalive=60)
    c.loop_start()
    print(f"[IMU原始模拟] 已连接 {args.broker}:{args.port}，模式={args.mode}，topic={args.topic}")

    rng = np.random.default_rng()
    i = 0
    try:
        while True:
            if args.mode == "alternate":
                mode = "fall" if (i // 4) % 2 == 0 else "normal"
            else:
                mode = args.mode

            window = make_window(mode, rng).tolist()
            payload = {
                "device": args.device,
                "fs": FS,
                "window": window,  # [[ax,ay,az], ...] 共 100 个采样
            }
            c.publish(args.topic, json.dumps(payload))
            print(f"[IMU原始模拟] 发布 {mode} 窗口 ({len(window)} 采样)")
            time.sleep(args.interval)
            i += 1
    except KeyboardInterrupt:
        pass
    c.disconnect()
    print("[IMU原始模拟] 已退出")


if __name__ == "__main__":
    main()
