"""
自动标注：用当前模型给采集的帧打框，生成 YOLO 标签，供人工修正后微调。

用法（项目根目录）：
  venv\\Scripts\\python.exe tools\\autolabel.py --input capture --conf 0.30

输出：<input>/labels/*.txt（与图片同名），每行 "class_id cx cy w h"（归一化）。
之后用 labelImg 打开 <input> 目录人工修正（删错框、补框）。
"""
import argparse
import os
import sys

import numpy as np
import torch
import cv2
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov5"))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(ROOT, "yolov5", "runs", "train", "fall_v4", "weights", "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="capture", help="采集帧目录")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.30, help="打框置信度阈值（低一点多框，人工删）")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = DetectMultiBackend(args.weights, device=dev, dnn=False, data=None, fp16=False)
    names = model.names

    img_dir = os.path.abspath(args.input)
    lbl_dir = os.path.join(img_dir, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    print(f"[自动标注] 共 {len(imgs)} 张图，阈值 {args.conf}，模型类别 {names}")

    n_box = 0
    for img_p in imgs:
        img0 = cv2.imread(img_p)
        h, w = img0.shape[:2]
        img = letterbox(img0, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(dev).float() / 255.0
        img = img.unsqueeze(0)
        det = non_max_suppression(model(img), args.conf, 0.45, max_det=50)[0]

        stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(lbl_dir, stem + ".txt")
        with open(lbl_p, "w") as f:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det.tolist():
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    n_box += 1

    print(f"[自动标注] 完成，共 {n_box} 个框，标签在 {lbl_dir}")
    print("下一步：用 labelImg 打开图片目录，人工修正错框/漏框。")


if __name__ == "__main__":
    main()
