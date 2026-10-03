"""
构建最终 2 类数据集：Roboflow 数据(重映射) + 原有数据，切分 train/val/test。

映射（6 类 -> 2 类）：
  falling(0)  -> down(1)    跌倒
  sleeping(2) -> down(1)    躺睡（横躺姿态，和跌倒视觉一致，宁可误报不可漏报）
  sitting(1)  -> normal(0)  坐
  standing(3) -> normal(0)  站
  wallking(4) -> normal(0)  走
  waving(5)   -> normal(0)  挥手

切分：Roboflow 数据 8:1:1 随机分 train/val/test；原有数据(你的2段视频)全部进 train，用于适配你的摄像头。
输出到 dataset_final/，不动原始 dataset 和 fall-1。
"""
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MAP = {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0}  # falling+sleeping -> down(1)

SRC_IMG = "fall-1/train/images"
SRC_LBL = "fall-1/train/labels"
ORIG_IMG = "dataset/images"
ORIG_LBL = "dataset/labels"
OUT = "dataset_final"

random.seed(42)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
for s in ("train", "val", "test"):
    os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)


def copy_label(src, dst, remap=None):
    if not os.path.exists(src):
        return 0
    n = 0
    with open(src) as fr, open(dst, "w") as fw:
        for line in fr:
            parts = line.split()
            if not parts:
                continue
            old = int(float(parts[0]))
            if remap is not None:
                if old not in remap:
                    continue
                parts[0] = str(remap[old])
            fw.write(" ".join(parts) + "\n")
            n += 1
    return n


# 1) Roboflow 数据：重映射 + 切分
rf_files = [f for f in os.listdir(SRC_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(rf_files)
n = len(rf_files)
n_val = int(n * 0.10)
n_test = int(n * 0.10)

down = normal = 0
for idx, f in enumerate(rf_files):
    if idx < n_val:
        split = "val"
    elif idx < n_val + n_test:
        split = "test"
    else:
        split = "train"
    stem = os.path.splitext(f)[0]
    shutil.copy2(os.path.join(SRC_IMG, f), f"{OUT}/images/{split}/{f}")
    nb = copy_label(os.path.join(SRC_LBL, stem + ".txt"),
                    f"{OUT}/labels/{split}/{stem}.txt", remap=MAP)
    # 统计
    for line in open(f"{OUT}/labels/{split}/{stem}.txt"):
        if line.split() and line.split()[0] == "1":
            down += 1
        elif line.split():
            normal += 1

# 2) 原有数据全部进 train（前缀 orig_ 避免重名）
for s in ("train", "val"):
    for f in os.listdir(f"{ORIG_IMG}/{s}"):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        dst = f"orig_{s}_{f}"
        shutil.copy2(f"{ORIG_IMG}/{s}/{f}", f"{OUT}/images/train/{dst}")
        copy_label(f"{ORIG_LBL}/{s}/{stem}.txt", f"{OUT}/labels/train/orig_{s}_{stem}.txt")

# 3) 统计
print("===== 最终数据集 dataset_final =====")
total = 0
for s in ("train", "val", "test"):
    imgs = len(os.listdir(f"{OUT}/images/{s}"))
    total += imgs
    print(f"  {s}: {imgs} 张图")
print(f"  合计: {total} 张图 | down 框 {down} | normal 框 {normal}")
print("完成。下一步写 data.yaml 并训练。")
