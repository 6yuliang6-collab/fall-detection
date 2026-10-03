"""
把人工修正后的采集数据合并进 dataset_final 的 train 集（前缀 cap_），供微调。

用法（labelImg 修正完之后，项目根目录）：
  venv\\Scripts\\python.exe tools\\merge_capture.py --input capture

会自动：
  - 把 capture/*.jpg 复制到 dataset_final/images/train/cap_*.jpg
  - 把 capture/labels/*.txt 复制到 dataset_final/labels/train/cap_*.txt（空标签也保留=背景帧）
"""
import argparse
import os
import shutil
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="capture")
    args = ap.parse_args()

    img_dir = os.path.abspath(args.input)
    lbl_dir = os.path.join(img_dir, "labels")
    out_img = os.path.join(ROOT, "dataset_final", "images", "train")
    out_lbl = os.path.join(ROOT, "dataset_final", "labels", "train")

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    n = 0
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        dst_name = f"cap_{stem}"
        shutil.copy2(p, os.path.join(out_img, dst_name + ".jpg"))
        lbl = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl):
            shutil.copy2(lbl, os.path.join(out_lbl, dst_name + ".txt"))
        else:
            # 没标签 = 背景帧，写个空标签
            open(os.path.join(out_lbl, dst_name + ".txt"), "w").close()
        n += 1

    print(f"[合并] 已把 {n} 张采集图合并进 dataset_final/train（前缀 cap_）")
    print("下一步：用 fall_v4 权重微调（见 README 或直接跑 train.py）")


if __name__ == "__main__":
    main()
