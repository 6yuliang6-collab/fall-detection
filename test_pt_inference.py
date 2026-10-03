import sys
sys.path.insert(0, 'inferlocal')

# 无 GUI 环境下把 imshow 系列变成空操作
import cv2
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda: None

from fallDownDetectYolo import FallDownDetectYolo

print("=== PT 路线测试：yolov5_best.pt ===")
detector = FallDownDetectYolo(
    yolov5_path='yolov5',
    weight_path='modelweight/yolov5_best.pt',
    image_path='yolov5/data/images/zidane.jpg',
    pt_or_onnx='pt'
)
detector.img_inference(save=True, save_path='result_pt.jpg')
print('[OK] PT inference finished')
