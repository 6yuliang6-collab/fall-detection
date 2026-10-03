import numpy as np
import cv2
import onnxruntime as ort


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


CLASS_NAMES = ['normal', 'down']

weight_path = 'modelweight/model.onnx'
image_path = 'yolov5/data/images/zidane.jpg'

print(f"[1] 加载 ONNX 模型: {weight_path}")
session = ort.InferenceSession(weight_path)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
print(f"    输入名: {input_name}, 输出名: {output_names}")

print(f"[2] 读取图片: {image_path}")
img0 = cv2.imread(image_path)
assert img0 is not None, "图片读取失败"
print(f"    原图尺寸: {img0.shape}")

print("[3] 预处理 (letterbox + 归一化)")
img, ratio, (dw, dh) = letterbox(img0, new_shape=(640, 640))
img_rgb = img[:, :, ::-1].astype(np.float32) / 255.0
img_nchw = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)
print(f"    输入张量: {img_nchw.shape} {img_nchw.dtype}")

print("[4] 推理")
outputs = session.run(output_names, {input_name: img_nchw})[0]
print(f"    原始输出 shape: {outputs.shape}")

print("[5] 后处理")
out = np.squeeze(outputs, axis=0)
boxes_xywh = out[:, :4]
confidences = out[:, 4]
class_scores = out[:, 5:]
class_ids = np.argmax(class_scores, axis=1)
confidences = confidences * class_scores[np.arange(len(class_scores)), class_ids]
boxes_xyxy = xywh_to_xyxy(boxes_xywh)

indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), 0.6, 0.3)
if len(indices) > 0:
    indices = np.array(indices).flatten()
else:
    indices = np.array([], dtype=int)

print(f"    NMS 后保留 {len(indices)} 个框")
for i in indices:
    conf = float(confidences[i])
    cid = int(class_ids[i])
    name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
    box = boxes_xyxy[i]
    # 坐标还原
    x1 = (box[0] - dw) / ratio[0]
    y1 = (box[1] - dh) / ratio[1]
    x2 = (box[2] - dw) / ratio[0]
    y2 = (box[3] - dh) / ratio[1]
    print(f"    -> class={name:6s} conf={conf:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

print("\n[OK] ONNX smoke test passed: model load + inference + postprocess all working")
