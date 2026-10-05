"""
云端 IMU 推理：加载随机森林模型，对原始加速度窗口做跌倒预测。

这就是"云端推理"的部分——不再由边缘端直接发 down/normal 结论，而是边缘端
发原始加速度窗口，云端提取 67 维特征并跑 RF，得到跌倒概率，再交给融合/警报。

用法：
  from imu_rf_inference import predict
  cls, prob = predict(window)   # window: [N,3] 的 Ax/Ay/Az 序列（g）
"""
import os

import joblib
import numpy as np

from imu_features import extract_features

_DIR = os.path.dirname(os.path.abspath(__file__))

_RF = joblib.load(os.path.join(_DIR, "imu_rf.joblib"))
_SCALER = joblib.load(os.path.join(_DIR, "imu_scaler.joblib"))
_FEATURE_NAMES = joblib.load(os.path.join(_DIR, "imu_feature_names.joblib"))

MODEL_LOADED = True
print(f"[IMU推理] 已加载随机森林模型（{len(_FEATURE_NAMES)} 特征，"
      f"{getattr(_RF, 'n_estimators', '?')} 棵树）")


def predict(window):
    """
    返回 (cls, prob)：
      cls  : 0=正常, 1=跌倒
      prob : 模型输出的「跌倒」概率 [0, 1]
    """
    feats = extract_features(window)
    x = _SCALER.transform(np.asarray([feats], dtype=np.float32))
    prob = float(_RF.predict_proba(x)[0][1])   # 第 2 类是跌倒(1)
    cls = 1 if prob >= 0.5 else 0
    return cls, prob


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    normal = np.column_stack([
        0.05 * rng.standard_normal(100),
        0.05 * rng.standard_normal(100),
        1.0 + 0.05 * rng.standard_normal(100),
    ])
    fall = np.column_stack([
        1.0 + 0.05 * rng.standard_normal(100),
        0.05 * rng.standard_normal(100),
        0.05 * rng.standard_normal(100),
    ])
    for name, w in [("正常", normal), ("跌倒", fall)]:
        cls, prob = predict(w)
        print(f"{name}窗口 -> 类别={cls}('down' if cls else 'normal') 跌倒概率={prob:.4f}")
