import cv2
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda: None

import sys
import runpy

sys.path.insert(0, 'inferlocal')

print("=== run.py CLI 测试 1: pt ===")
sys.argv = ['run.py', '--yolov5_path', 'yolov5',
            '--weight_path', 'modelweight/yolov5_best.pt',
            '--image_path', 'yolov5/data/images/zidane.jpg',
            '--model_type', 'pt', '--input_type', 'image',
            '--save', '--save_path', 'result_cli_pt.jpg']
runpy.run_path('inferlocal/run.py', run_name='__main__')

print("\n=== run.py CLI 测试 2: onnx ===")
sys.argv = ['run.py', '--yolov5_path', 'yolov5',
            '--weight_path', 'modelweight/model.onnx',
            '--image_path', 'yolov5/data/images/zidane.jpg',
            '--model_type', 'onnx', '--input_type', 'image',
            '--save', '--save_path', 'result_cli_onnx.jpg']
runpy.run_path('inferlocal/run.py', run_name='__main__')

print('\n[OK] run.py CLI tests finished')
