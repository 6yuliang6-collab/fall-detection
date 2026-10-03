"""
视频抽帧工具：从录制的视频中提取训练图片。

用法（在项目根目录执行）：
  venv\\Scripts\\python.exe tools\\video2frames.py <视频路径> <输出目录> [--interval N]

示例：
  venv\\Scripts\\python.exe tools\\video2frames.py D:\\video\\fall.mp4 D:\\frames\\fall --interval 5
  （每 5 帧抽 1 张，输出到 D:\\frames\\fall）
"""
import cv2
import os
import argparse


def main():
    ap = argparse.ArgumentParser(description="视频抽帧工具")
    ap.add_argument('video', help='视频文件路径')
    ap.add_argument('outdir', help='输出图片目录')
    ap.add_argument('--interval', type=int, default=5, help='每隔多少帧抽 1 帧（默认 5）')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('无法打开视频:', args.video)
        return

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.interval == 0:
            name = os.path.join(args.outdir, 'frame_{:06d}.jpg'.format(idx))
            cv2.imwrite(name, frame)
            saved += 1
        idx += 1

    cap.release()
    print('完成: 视频共 {} 帧, 抽取 {} 张到 {}'.format(idx, saved, args.outdir))


if __name__ == '__main__':
    main()
