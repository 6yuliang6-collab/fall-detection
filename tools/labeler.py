"""
轻量标注工具（cv2 实现，替代 labelImg，稳定不闪退）。

用法（项目根目录）：
  venv\\Scripts\\python.exe tools\\labeler.py --input capture

操作：
  左键拖拽        画一个新框（用当前类别）
  左键单击已有框   选中它（黄色高亮）
    按 0 / 1      把选中框改成 normal / down（也同时设定"新框"的类别）
    按 x          删除选中框
  a / d           上一张 / 下一张（自动保存当前张）
  s               保存当前张
  q               退出（自动保存）

类别：0 = normal(绿) ，1 = down(红)
"""
import argparse
import os
import sys
import glob

import cv2

CLASSES = ["normal", "down"]
COLORS = [(0, 255, 0), (0, 0, 255)]  # 0=绿, 1=红


class Labeler:
    def __init__(self, img_dir):
        self.img_dir = os.path.abspath(img_dir)
        self.lbl_dir = os.path.join(self.img_dir, "labels")
        os.makedirs(self.lbl_dir, exist_ok=True)
        self.imgs = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                           glob.glob(os.path.join(self.img_dir, "*.png")))
        if not self.imgs:
            print(f"[!] {img_dir} 里没有图片")
            sys.exit(1)
        self.idx = 0
        self.cur_class = 1       # 默认 down（因为多数误标是 down，要改成 normal）
        self.boxes = []          # [x1,y1,x2,y2,cls]
        self.selected = -1
        self.drawing = False
        self.start = None
        self.img = None
        self.h = self.w = 0

    def _stem(self):
        return os.path.splitext(os.path.basename(self.imgs[self.idx]))[0]

    def load(self):
        self.img = cv2.imread(self.imgs[self.idx])
        self.h, self.w = self.img.shape[:2]
        self.boxes = []
        self.selected = -1
        lbl = os.path.join(self.lbl_dir, self._stem() + ".txt")
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) == 5:
                    c = int(float(p[0]))
                    cx, cy, bw, bh = map(float, p[1:])
                    x1 = int((cx - bw / 2) * self.w)
                    y1 = int((cy - bh / 2) * self.h)
                    x2 = int((cx + bw / 2) * self.w)
                    y2 = int((cy + bh / 2) * self.h)
                    self.boxes.append([x1, y1, x2, y2, c])

    def save(self):
        with open(os.path.join(self.lbl_dir, self._stem() + ".txt"), "w") as f:
            for x1, y1, x2, y2, c in self.boxes:
                cx = (x1 + x2) / 2 / self.w
                cy = (y1 + y2) / 2 / self.h
                bw = (x2 - x1) / self.w
                bh = (y2 - y1) / self.h
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 先判断是否点中已有框
            for i, (x1, y1, x2, y2, c) in enumerate(self.boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected = i
                    self.drawing = False
                    return
            self.selected = -1
            self.drawing = True
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            x1, y1 = self.start
            if x - x1 > 6 and y - y1 > 6:
                self.boxes.append([x1, y1, x, y, self.cur_class])
            self.drawing = False

    def draw(self):
        disp = self.img.copy()
        for i, (x1, y1, x2, y2, c) in enumerate(self.boxes):
            color = (0, 255, 255) if i == self.selected else COLORS[c]
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, CLASSES[c], (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # 顶部信息条
        cv2.rectangle(disp, (0, 0), (self.w, 64), (40, 40, 40), -1)
        cv2.putText(disp, f"{self.idx + 1}/{len(self.imgs)}  {self._stem()}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(disp, f"当前类别={CLASSES[self.cur_class]}   [拖拽画框 | 单击选中 | 0/1改类 | x删 | a/d换图 | s存 | q退]",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return disp

    def run(self):
        cv2.namedWindow("labeler", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("labeler", self.on_mouse)
        self.load()
        while True:
            cv2.imshow("labeler", self.draw())
            k = cv2.waitKey(20) & 0xFF
            if k == ord("q"):
                self.save()
                break
            elif k == ord("s"):
                self.save()
                print(f"已保存 {self._stem()}")
            elif k == ord("0"):
                self.cur_class = 0
                if self.selected >= 0:
                    self.boxes[self.selected][4] = 0
            elif k == ord("1"):
                self.cur_class = 1
                if self.selected >= 0:
                    self.boxes[self.selected][4] = 1
            elif k in (ord("x"), ord("X")):
                if self.selected >= 0:
                    self.boxes.pop(self.selected)
                    self.selected = -1
            elif k in (ord("d"), ord("D"), 83):   # 下一张
                self.save()
                self.idx = min(self.idx + 1, len(self.imgs) - 1)
                self.load()
            elif k in (ord("a"), ord("A"), 81):   # 上一张
                self.save()
                self.idx = max(self.idx - 1, 0)
                self.load()
        cv2.destroyAllWindows()
        print("标注结束，标签已保存在 capture/labels/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="capture")
    Labeler(ap.parse_args().input).run()
