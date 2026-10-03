"""
训练集/验证集划分工具：把标注好的图片分成 train 和 val。

用法（在项目根目录执行）：
  venv\\Scripts\\python.exe tools\\split_train_val.py <图片目录> <标签目录> [--val 0.2] [--group-by]

参数：
  --val 0.2        验证集比例（默认 0.2）
  --group-by       按"视频/序列"分组切分（强烈推荐！），而不是按单张图片随机切分
  --group-regex    自定义分组正则，默认把文件名末尾的 -帧号 去掉作为视频名
  --seed 0         随机种子

为什么需要 --group-by：
  同一段视频连续抽出的帧几乎一模一样（fall-01-...-001 和 -002 只差几毫秒）。
  如果按"单张图片"随机切分，这些近重复帧会同时进入 train 和 val，
  导致验证集"作弊"（模型早就在 train 里见过几乎相同的图），
  验证指标虚高、无法反映真实泛化能力。按视频分组切分可避免这个问题。
"""
import os
import random
import re
import shutil
import argparse

IMG_EXT = ('.jpg', '.jpeg', '.png')


def group_key(fname, group_regex):
    """返回该图片所属的视频/序列名（用于分组切分）。"""
    if group_regex:
        m = re.search(group_regex, fname)
        return m.group(0) if m else fname
    # 默认：去掉文件名末尾的 -帧号.扩展名，例如 fall-01-cam0-rgb-001.jpg -> fall-01-cam0-rgb
    return re.sub(r'-\d+\.[^.]+$', '', fname)


def main():
    ap = argparse.ArgumentParser(description="划分训练集/验证集（支持按视频分组避免泄漏）")
    ap.add_argument('imgdir', help='标注好的图片目录')
    ap.add_argument('lbldir', help='对应的标签目录（.txt，与图片同名）')
    ap.add_argument('--val', type=float, default=0.2, help='验证集比例（默认 0.2）')
    ap.add_argument('--seed', type=int, default=0, help='随机种子')
    ap.add_argument('--group-by', action='store_true', help='按视频/序列分组切分（推荐）')
    ap.add_argument('--group-regex', type=str, default=None,
                    help='分组正则（默认去掉末尾 -帧号 作为视频名）')
    args = ap.parse_args()

    random.seed(args.seed)
    imgs = [f for f in os.listdir(args.imgdir) if f.lower().endswith(IMG_EXT)]
    if not imgs:
        print('图片目录里没有图片:', args.imgdir)
        return

    if args.group_by:
        # 按视频/序列分组，整组进入 train 或 val
        groups = {}
        for f in imgs:
            groups.setdefault(group_key(f, args.group_regex), []).append(f)
        n_val = max(1, int(len(groups) * args.val))
        keys = sorted(groups.keys())
        random.shuffle(keys)
        val_set = set()
        for k in keys[:n_val]:
            val_set.update(groups[k])
        print('按视频分组切分：共 {} 段视频，{} 段进 val'.format(len(groups), n_val))
        if len(groups) < 3:
            print('[警告] 视频段太少（{} 段）！分组切分后 val 可能只含 1 段视频，'
                  '代表性有限，请尽量采集更多视频。'.format(len(groups)))
    else:
        # 原逻辑：按单张图片随机切分（可能造成近重复帧泄漏，不推荐）
        print('[警告] 未启用 --group-by，按单张图片切分可能导致同一视频的近重复帧'
              '同时进入 train/val，验证指标虚高。建议加 --group-by。')
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * args.val))
        val_set = set(imgs[:n_val])

    out_train = 'dataset/images/train'
    out_val = 'dataset/images/val'
    lbl_train = 'dataset/labels/train'
    lbl_val = 'dataset/labels/val'
    for d in (out_train, out_val, lbl_train, lbl_val):
        os.makedirs(d, exist_ok=True)

    n_tr = n_va = 0
    for f in imgs:
        split = 'val' if f in val_set else 'train'
        stem = os.path.splitext(f)[0]
        shutil.copy2(os.path.join(args.imgdir, f),
                     'dataset/images/{}/{}.{}'.format(split, stem, f.split('.')[-1]))
        lbl = os.path.join(args.lbldir, stem + '.txt')
        if os.path.exists(lbl):
            shutil.copy2(lbl, 'dataset/labels/{}/{}.txt'.format(split, stem))
        if split == 'val':
            n_va += 1
        else:
            n_tr += 1

    print('完成: 训练集 {} 张, 验证集 {} 张'.format(n_tr, n_va))
    print('已复制到 dataset/images/train|val 和 dataset/labels/train|val')


if __name__ == '__main__':
    main()
