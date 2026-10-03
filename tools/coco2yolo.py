"""
COCO JSON -> YOLO txt 转换工具

用法（在项目根目录执行）：
  venv\\Scripts\\python.exe tools\\coco2yolo.py <标注.json> <输出标签目录>

说明：
  COCO 的 bbox 是像素坐标 [x, y, w, h]（左上角 x, y, 宽, 高）
  YOLO 需要的是归一化坐标 [class_id, x_center, y_center, width, height]（0~1）
"""
import json
import os
import argparse


def main():
    ap = argparse.ArgumentParser(description="COCO JSON 转 YOLO txt")
    ap.add_argument('json_path', help='COCO 格式的标注 json 文件')
    ap.add_argument('out_dir', help='输出 YOLO 标签目录')
    args = ap.parse_args()

    with open(args.json_path, encoding='utf-8') as f:
        coco = json.load(f)

    # 检查是否是标准 COCO 格式
    if 'images' not in coco or 'annotations' not in coco or 'categories' not in coco:
        print('[错误] 这不是标准 COCO 格式，缺少 images/annotations/categories 字段')
        print('请把 json 文件路径发给我，我看看你的实际格式再调整脚本')
        return

    # 图片信息: id -> (文件名, 宽, 高)
    img_info = {img['id']: (img['file_name'], img['width'], img['height'])
                for img in coco['images']}

    # 类别映射: category_id -> 从 0 开始的序号（按 categories 里的顺序）
    cat_id_map = {c['id']: i for i, c in enumerate(coco['categories'])}

    # 按图片分组标注
    ann_by_img = {}
    for ann in coco['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for img_id, anns in ann_by_img.items():
        fname, w, h = img_info[img_id]
        stem = os.path.splitext(os.path.basename(fname))[0]
        with open(os.path.join(args.out_dir, stem + '.txt'), 'w') as f:
            for ann in anns:
                x, y, bw, bh = ann['bbox']
                cls = cat_id_map[ann['category_id']]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(cls, cx, cy, nw, nh))
        n += 1

    # 生成 classes.txt（类别顺序，方便核对）
    with open(os.path.join(args.out_dir, 'classes.txt'), 'w') as f:
        for c in coco['categories']:
            f.write(c['name'] + '\n')

    print('转换完成: {} 张图片的标签已写入 {}'.format(n, args.out_dir))
    print('类别顺序: {}'.format([c['name'] for c in coco['categories']]))
    print('请确认类别顺序是 [normal, down]，即 0=normal, 1=down')


if __name__ == '__main__':
    main()
