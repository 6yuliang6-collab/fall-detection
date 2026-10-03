"""
生成 IMU 模态展示图：RF vs 1D-CNN 对比柱状图 + 混淆矩阵，直接贴进 PPT。

输出：
  analysis/imu_comparison.png   两模型 P/R/F1 对比柱状图
  analysis/imu_confusion.png    两模型混淆矩阵（展示误报/漏报）
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "_imu", "augmented_dataset.csv")
OUT = os.path.join(ROOT, "analysis")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class CNN1D(nn.Module):
    def __init__(self, n_feat=67):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).squeeze(-1))


def main():
    df = pd.read_csv(CSV)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

    # RF
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
    yp_rf = rf.predict(X_te_s)
    # 1D-CNN
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = CNN1D().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X_tr_s).unsqueeze(1).to(dev)
    yt = torch.from_numpy(y_tr).to(dev)
    Xv = torch.from_numpy(X_te_s).unsqueeze(1).to(dev)
    n, bs = len(Xt), 256
    for _ in range(30):
        m.train()
        for i in range(0, n, bs):
            idx = torch.randperm(n)[i:i + bs]
            opt.zero_grad()
            lf(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        yp_cnn = m(Xv).argmax(1).cpu().numpy()

    models = {"随机森林 RF": yp_rf, "1D-CNN": yp_cnn}
    colors = {"随机森林 RF": "#4c78a8", "1D-CNN": "#f58518"}

    # 图1：P/R/F1 对比柱状图
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(3)
    width = 0.35
    for i, (name, yp) in enumerate(models.items()):
        p, r, f, _ = precision_recall_fscore_support(y_te, yp, labels=[1], zero_division=0)
        vals = [p[0], r[0], f[0]]
        ax.bar(x + i * width, vals, width, label=name, color=colors[name])
        for xi, v in zip(x + i * width, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x + width / 2, ["精确率 P", "召回率 R", "F1"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("分数")
    ax.set_title("IMU 模态：RF vs 1D-CNN（跌倒类）")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "imu_comparison.png"), dpi=130)
    plt.close()

    # 图2：混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, (name, yp) in zip(axes, models.items()):
        cm = confusion_matrix(y_te, yp, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], ["非跌倒", "跌倒"])
        ax.set_yticks([0, 1], ["非跌倒", "跌倒"])
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "imu_confusion.png"), dpi=130)
    plt.close()

    print("已生成:")
    print("  ", os.path.join(OUT, "imu_comparison.png"))
    print("  ", os.path.join(OUT, "imu_confusion.png"))


if __name__ == "__main__":
    main()
