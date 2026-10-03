import sys
sys.path.insert(0, 'inferlocal')

import cv2
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda: None

from fallDownDetectYolo import FallDownDetectYolo

print("=== ONNX 路线测试（模块）: model.onnx ===")
detector = FallDownDetectYolo(
    yolov5_path='yolov5',
    weight_path='modelweight/model.onnx',
    image_path='yolov5/data/images/zidane.jpg',
    pt_or_onnx='onnx'
)
detector.img_inference(save=True, save_path='result_onnx_module.jpg')
print('[OK] ONNX module inference finished')
