"""
误报 vs 漏报 分析：在保留测试集上扫描置信度阈值，输出 P/R/F1 曲线。

原理：
  - 阈值调低 -> 检出更多（召回 R 高、漏报少），但误报(FP)多、精确率 P 低
  - 阈值调高 -> 误报少、P 高，但漏报多、R 低
  这就是"误报 vs 漏报"的权衡，用曲线量化出来，供报告讨论。

输出：
  analysis/false_alarm_curves.png   阈值 vs P/R/F1 + P-R 曲线
  analysis/threshold_metrics.csv    每个阈值下的指标（可直接写进报告）

用法（项目根目录）：
  venv\\Scripts\\python.exe tools\\false_alarm_analysis.py
"""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "yolov5", "runs", "train", "fall_v4", "weights", "best.pt")
TEST_IMG = os.path.join(ROOT, "dataset_final", "images", "test")
OUT_DIR = os.path.join(ROOT, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-6)


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(WEIGHTS, device=dev, dnn=False, data=None, fp16=False)
    print(f"[分析] 模型已加载，device={dev}")

    all_preds = []   # (img_idx, conf, cls, [x1,y1,x2,y2])
    all_gts = {}     # img_idx -> [(cls, [x1,y1,x2,y2]), ...]

    test_imgs = sorted(glob.glob(os.path.join(TEST_IMG, "*.jpg")))
    for idx, img_p in enumerate(test_imgs):
        img0 = cv2.imread(img_p)
        h, w = img0.shape[:2]

        # 真值
        lbl_p = img_p.replace("images", "labels").replace(".jpg", ".txt")
        gts = []
        if os.path.exists(lbl_p):
            for line in open(lbl_p):
                p = line.split()
                if len(p) != 5:
                    continue
                cls = int(float(p[0]))
                cx, cy, bw, bh = map(float, p[1:])
                gts.append((cls, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h]))
        all_gts[idx] = gts

        # 预测（conf 极低，拿到所有框，后面按阈值过滤）
        img = letterbox(img0, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(dev).float() / 255.0
        img = img.unsqueeze(0)
        det = non_max_suppression(model(img), 0.001, 0.45, max_det=300)[0]
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                all_preds.append((idx, float(conf), int(cls), xyxy))

    print(f"[分析] 共 {len(test_imgs)} 张测试图，{len(all_preds)} 个预测框，"
          f"{sum(len(v) for v in all_gts.values())} 个真值框")

    # 扫描阈值，算 P/R/F1（整体 + 每类）
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for th in thresholds:
        # 整体
        tp = fp = fn = 0
        tp_c = {c: 0 for c in (0, 1)}
        fp_c = {c: 0 for c in (0, 1)}
        fn_c = {c: 0 for c in (0, 1)}
        matched = {i: set() for i in all_gts}

        for idx, conf, cls, box in all_preds:
            if conf < th:
                continue
            best_iou, best_j = 0.0, -1
            for j, (gcls, gbox) in enumerate(all_gts[idx]):
                if gcls != cls or j in matched[idx]:
                    continue
                i = iou(box, gbox)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= 0.5:
                tp += 1
                tp_c[cls] += 1
                matched[idx].add(best_j)
            else:
                fp += 1
                fp_c[cls] += 1

        for i in all_gts:
            for j, (gcls, _) in enumerate(all_gts[i]):
                if j not in matched[i]:
                    fn += 1
                    fn_c[gcls] += 1

        def prf(t, f):
            P = t / (t + f[0]) if t + f[0] else 0.0
            R = t / (t + f[1]) if t + f[1] else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            return P, R, F

        P, R, F = prf(tp, (fp, fn))
        Pd, Rd, Fd = prf(tp_c[1], (fp_c[1], fn_c[1]))
        rows.append((th, P, R, F, Pd, Rd, Fd, tp_c[1], fp_c[1], fn_c[1]))

    # 找最优阈值（整体 F1 最大）
    best = max(rows, key=lambda r: r[3])
    print(f"\n[分析] 最优阈值（整体 F1 最大）: conf={best[0]:.2f}  "
          f"P={best[1]:.3f} R={best[2]:.3f} F1={best[3]:.3f}")
    # 当前系统阈值 0.25 的指标
    cur = [r for r in rows if abs(r[0] - 0.25) < 0.001]
    if cur:
        c = cur[0]
        print(f"[分析] 当前阈值 0.25 的指标: 整体 P={c[1]:.3f} R={c[2]:.3f} F1={c[3]:.3f}"
              f" | down P={c[4]:.3f} R={c[5]:.3f} F1={c[6]:.3f}")

    # 存 CSV
    csv_path = os.path.join(OUT_DIR, "threshold_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("conf_thresh,all_P,all_R,all_F1,down_P,down_R,down_F1,down_TP,down_FP,down_FN\n")
        for r in rows:
            f.write(",".join(f"{x:.4f}" for x in r) + "\n")

    # 画图
    th = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(th, [r[1] for r in rows], "o-", label="Precision (精确率)")
    ax.plot(th, [r[2] for r in rows], "s-", label="Recall (召回率)")
    ax.plot(th, [r[3] for r in rows], "^-", label="F1", linewidth=2)
    ax.axvline(best[0], color="gray", linestyle="--", label=f"最优阈值 {best[0]:.2f}")
    ax.set_xlabel("置信度阈值 conf")
    ax.set_ylabel("分数")
    ax.set_title("整体 P / R / F1 vs 阈值")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(th, [r[4] for r in rows], "o-", label="down 精确率 P")
    ax.plot(th, [r[5] for r in rows], "s-", label="down 召回率 R")
    ax.plot(th, [r[6] for r in rows], "^-", label="down F1", linewidth=2)
    ax.axvline(best[0], color="gray", linestyle="--", label=f"最优阈值 {best[0]:.2f}")
    ax.set_xlabel("置信度阈值 conf")
    ax.set_ylabel("分数")
    ax.set_title("跌倒(down)类 P / R / F1 vs 阈值（误报 vs 漏报核心）")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "false_alarm_curves.png")
    plt.savefig(fig_path, dpi=130)
    print(f"\n[分析] 曲线已保存: {fig_path}")
    print(f"[分析] 指标表已保存: {csv_path}")


if __name__ == "__main__":
    main()
