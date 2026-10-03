"""
IMU 模态：随机森林(RF) vs 1D-CNN 对比。

数据：SisFall 公开 IMU 跌倒数据集的统计特征（9000 样本 × 67 维特征 + 标签），
      特征来自加速度计/陀螺仪窗口的统计量（均值/方差/峰度/偏度/相关系数/姿态角等）。
      标签：0=日常活动(ADL/非跌倒)，1=跌倒。

模型：
  - RF     : 经典机器学习，直接吃 67 维手工特征
  - 1D-CNN : 深度学习，把 67 维特征当一维序列，用一维卷积学特征表示

输出：各自在保留 test 集上的 P/R/F1 + 混淆矩阵，写进 analysis/。

用法（项目根目录）：
  venv\\Scripts\\python.exe tools\\imu_rf_cnn.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "_imu", "augmented_dataset.csv")
OUT = os.path.join(ROOT, "analysis", "imu_rf_cnn_results.txt")


class CNN1D(nn.Module):
    """小型 1D-CNN：把 67 维特征当作长度 67 的一维信号，用卷积学表示。"""
    def __init__(self, n_feat=67):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):  # x: [B, 1, n_feat]
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def main():
    df = pd.read_csv(CSV)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    n_feat = X.shape[1]
    print(f"[IMU] 数据: {len(df)} 样本 × {n_feat} 特征，标签分布 {dict(zip(*np.unique(y, return_counts=True)))}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    results = []

    # ---------- 1) 随机森林 ----------
    print("\n[RF] 训练随机森林(200 棵树)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    yp_rf = rf.predict(X_te_s)
    p, r, f, _ = precision_recall_fscore_support(y_te, yp_rf, labels=[0, 1], zero_division=0)
    results.append(("随机森林 RF", p, r, f))
    print(f"[RF] 完成。整体准确率 {np.mean(yp_rf == y_te):.4f}")

    # ---------- 2) 1D-CNN ----------
    print("\n[CNN] 训练 1D-CNN...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN1D(n_feat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X_tr_s).unsqueeze(1).to(device)  # [N, 1, F]
    yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_te_s).unsqueeze(1).to(device)
    n = len(Xt)
    bs = 256
    EPOCHS = 30
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1}/{EPOCHS}  loss={tot / n:.4f}")
    model.eval()
    with torch.no_grad():
        yp_cnn = model(Xv).argmax(1).cpu().numpy()
    p, r, f, _ = precision_recall_fscore_support(y_te, yp_cnn, labels=[0, 1], zero_division=0)
    results.append(("1D-CNN", p, r, f))
    print(f"[CNN] 完成。整体准确率 {np.mean(yp_cnn == y_te):.4f}")

    # ---------- 汇总输出 ----------
    names = ["非跌倒(0)", "跌倒(1)", "宏平均"]
    lines = []
    lines.append("=" * 60)
    lines.append("IMU 模态模型对比：随机森林 RF vs 1D-CNN")
    lines.append(f"数据: SisFall 特征集，test 集 {len(y_te)} 样本")
    lines.append("=" * 60)
    lines.append(f"{'模型':<14}{'类别':<12}{'P':>8}{'R':>8}{'F1':>8}")
    for model_name, p, r, f in results:
        for i, nm in enumerate(names):
            if i == 2:
                pv, rv, fv = p.mean(), r.mean(), f.mean()
            else:
                pv, rv, fv = p[i], r[i], f[i]
            lines.append(f"{model_name:<14}{nm:<12}{pv:>8.4f}{rv:>8.4f}{fv:>8.4f}")
        lines.append("-" * 60)
    lines.append("")
    lines.append("结论：跌倒(1)类的 F1 是关键指标，比较两个模型在跌倒类上的表现。")
    txt = "\n".join(lines)
    print("\n" + txt)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"\n结果已保存: {OUT}")


if __name__ == "__main__":
    main()
