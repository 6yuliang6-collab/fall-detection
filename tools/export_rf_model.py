"""
导出随机森林(RF)模型，供云端推理使用。

背景：tools/imu_rf_cnn.py 训练完 RF 只写了文本结果，没保存模型。
本脚本用同一份数据、同样的 70/30 切分 + random_state=42，重新训练 RF，
并把模型、标准化器和特征名保存到 system/ 下，云端直接加载。

产出：
  system/imu_rf.joblib            # RandomForestClassifier
  system/imu_scaler.joblib        # StandardScaler（训练集上拟合）
  system/imu_feature_names.joblib # 67 个特征列名（按 CSV 原始顺序）

用法（项目根目录）：
  venv\\Scripts\\python.exe tools\\export_rf_model.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "_imu", "augmented_dataset.csv")
SYSTEM = os.path.join(ROOT, "system")


def main():
    df = pd.read_csv(CSV)
    feat_cols = [c for c in df.columns if c != "label"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    print(f"[导出] 数据: {len(df)} 样本 × {X.shape[1]} 特征")

    # 与 imu_rf_cnn.py 完全一致：70/30、stratify、random_state=42
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)

    yp = rf.predict(X_te_s)
    p, r, f, _ = precision_recall_fscore_support(
        y_te, yp, labels=[0, 1], zero_division=0)
    print(f"[导出] test 集：非跌倒 P={p[0]:.4f} R={r[0]:.4f} F1={f[0]:.4f}  |  "
          f"跌倒 P={p[1]:.4f} R={r[1]:.4f} F1={f[1]:.4f}")

    os.makedirs(SYSTEM, exist_ok=True)
    joblib.dump(rf, os.path.join(SYSTEM, "imu_rf.joblib"))
    joblib.dump(scaler, os.path.join(SYSTEM, "imu_scaler.joblib"))
    joblib.dump(feat_cols, os.path.join(SYSTEM, "imu_feature_names.joblib"))
    print(f"[导出] 已保存模型到 {SYSTEM}")
    print(f"        imu_rf.joblib / imu_scaler.joblib / imu_feature_names.joblib")


if __name__ == "__main__":
    main()
