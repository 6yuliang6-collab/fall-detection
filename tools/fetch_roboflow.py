"""
从 Roboflow 下载公开跌倒数据集，重映射到本项目 2 类格式，并合并进 dataset。

用法（在项目根目录执行）：
  venv\\Scripts\\python.exe tools\\fetch_roboflow.py --api-key 你的key --workspace go-ygbsj --project fall-awbxa --version 1

会自动：
  1. 下载 YOLOv5 格式数据集到 _rf_data/
  2. 读取其 data.yaml，把类别名映射到 2 类：0=normal(站立/行走/坐等), 1=down(跌倒/躺倒)
  3. 重写标签，并打印映射关系供你核对
  4. 合并到 dataset_merged/（保留原有 dataset 不动）
"""
import argparse
import os
import shutil
import sys

# 跌倒类关键词（匹配到就归为 down=1，其余归 normal=0）
FALL_KEYWORDS = ["fall", "down", "lying", "lie", "fallen", "ground", "drop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--workspace", default="go-ygbsj")
    ap.add_argument("--project", default="fall-awbxa")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    from roboflow import Roboflow
    rf = Roboflow(api_key=args.api_key)
    proj = rf.workspace(args.workspace).project(args.project)
    print(f"[1/4] 下载数据集 {args.workspace}/{args.project} v{args.version} ...")
    ds = proj.version(args.version).download("yolov5")
    ds_path = ds.location
    print("     下载到:", ds_path)

    # 读取类别名
    data_yaml = os.path.join(ds_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print("[!] 找不到 data.yaml，下载结构异常")
        return
    import yaml
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    print("[2/4] 原始类别:", names)

    # 建立映射：原 class id -> 新 class id (0=normal, 1=down)
    mapping = {}
    for i, n in enumerate(names):
        low = str(n).lower()
        mapping[i] = 1 if any(k in low for k in FALL_KEYWORDS) else 0
    print("      映射关系:", {names[i]: ("down" if mapping[i] else "normal") for i in range(len(names))})

    # 扫描 label 文件并重映射
    out_img = "dataset_merged/images"
    out_lbl = "dataset_merged/labels"
    n_copied = 0
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(ds_path, split, "images")
        lbl_dir = os.path.join(ds_path, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            stem, ext = os.path.splitext(f)
            lbl = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl):
                continue
            # 图片
            os.makedirs(os.path.join(out_img, split), exist_ok=True)
            shutil.copy2(os.path.join(img_dir, f), os.path.join(out_img, split, f))
            # 标签重映射
            os.makedirs(os.path.join(out_lbl, split), exist_ok=True)
            with open(lbl) as fr, open(os.path.join(out_lbl, split, stem + ".txt"), "w") as fw:
                for line in fr:
                    parts = line.split()
                    if not parts:
                        continue
                    old_cls = int(float(parts[0]))
                    new_cls = mapping.get(old_cls)
                    if new_cls is None:
                        continue
                    parts[0] = str(new_cls)
                    fw.write(" ".join(parts) + "\n")
            n_copied += 1

    print(f"[3/4] 已合并 {n_copied} 张图到 dataset_merged/（train/valid/test）")
    print("[4/4] 完成。下一步：把 dataset_merged 与原有 dataset 合并后重新切分、训练。")


if __name__ == "__main__":
    main()
