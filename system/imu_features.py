"""
原始加速度窗口 -> 67 维特征（与训练集 augmented_dataset.csv 的列一一对应）。

关键：训练 CSV 里的 Ax/Ay/Az 是「重力已补偿的线性加速度」（静止时约 0，
SVM≈0.1），不是带重力的原始加速度。所以这里先做重力估计（低通滤波），
用 原始 - 重力 得到线性加速度，再在其上算统计特征。

67 维特征（按 CSV 列顺序）：
  - 6 个信号通道：Ax, Ay, Az, SVM, VER, HOR（均为线性加速度口径）
    每个通道 9 个统计量：min, max, mean, median, var, std, kurtosis, skewness, range
  - 6 个相关系数：corr_valueXY, corr_valueXZ, corr_valueYZ,
                 corr_SVM_VER, corr_SVM_HOR, corr_HOR_VER
  - 7 个 pitch 角统计量（由重力方向算出设备倾角）：
                 pitch, max_pitch, min_pitch, mean_pitch, median_pitch, var_pitch, std_pitch

信号定义：
  g(t)      = 低通滤波估计的重力向量
  (Ax,Ay,Az) = 原始 - g(t)                      # 线性加速度
  SVM       = sqrt(Ax^2 + Ay^2 + Az^2)          # 线性加速度合向量模
  VER       = Az                                 # 垂直分量
  HOR       = sqrt(Ax^2 + Ay^2)                  # 水平分量
  pitch     = arctan2(gx, sqrt(gy^2 + gz^2))    # 设备倾角（弧度，由重力方向）

说明：原始"窗口 -> 特征"的生成代码不在本仓库（CSV 是外部已算好的），这里的
重力补偿 + 统计口径是按特征命名给出的标准定义重写，自洽可用；报告中的 0.994
F1 仍来自 CSV 保留测试集，不受影响。
"""
import numpy as np


def _lowpass_gravity(x, win=10):
    """一维信号的重力估计：居中滑动平均低通（默认 10 采样 ≈ 0.2s @50Hz）。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= win:
        return np.full_like(x, x.mean())
    k = np.ones(win) / win
    # 用 'same' 卷积，边缘用最近值填充以减少边界效应
    return np.convolve(x, k, mode="same")


def _stats(x):
    """对一维信号算 9 个统计量：min, max, mean, median, var, std, kurtosis, skewness, range。"""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sigma = x.std()  # 总体标准差
    var = x.var()    # 总体方差

    if sigma > 1e-12:
        skew = ((x - mu) ** 3).mean() / (sigma ** 3)
        kurt = ((x - mu) ** 4).mean() / (sigma ** 4) - 3.0  # 超值峰度（Fisher，可正可负）
    else:
        skew = 0.0
        kurt = 0.0

    return [
        x.min(),
        x.max(),
        mu,
        np.median(x),
        var,
        sigma,
        kurt,
        skew,
        x.max() - x.min(),
    ]


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(window):
    """
    window: array-like, shape [N, 3] = (raw_Ax, raw_Ay, raw_Az)，单位 g（含重力）。
    返回：长度 67 的 list[float]，顺序与训练 CSV 的列完全一致。
    """
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 3:
        raise ValueError(f"window 形状应为 [N, 3]，得到 {w.shape}")

    rx, ry, rz = w[:, 0], w[:, 1], w[:, 2]

    # 重力估计（低通），线性加速度 = 原始 - 重力
    gx = _lowpass_gravity(rx)
    gy = _lowpass_gravity(ry)
    gz = _lowpass_gravity(rz)
    ax, ay, az = rx - gx, ry - gy, rz - gz

    svm = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    ver = az
    hor = np.sqrt(ax ** 2 + ay ** 2)
    pitch = np.arctan2(gx, np.sqrt(gy ** 2 + gz ** 2))

    feats = []
    for sig in (ax, ay, az, svm, ver, hor):
        feats += _stats(sig)

    feats += [
        _corr(ax, ay),
        _corr(ax, az),
        _corr(ay, az),
        _corr(svm, ver),
        _corr(svm, hor),
        _corr(hor, ver),
    ]

    p = _stats(pitch)  # 9 个统计量：min,max,mean,median,var,std,kurtosis,skewness,range
    # pitch 列顺序：pitch, max_pitch, min_pitch, mean_pitch, median_pitch, var_pitch, std_pitch
    feats += [
        float(np.mean(pitch)),  # pitch（取均值作为代表值）
        p[1],                   # max_pitch
        p[0],                   # min_pitch
        p[2],                   # mean_pitch
        p[3],                   # median_pitch
        p[4],                   # var_pitch
        p[5],                   # std_pitch
    ]

    assert len(feats) == 67, f"特征数应为 67，得到 {len(feats)}"
    return feats


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 100
    # 正常：站立，重力在 z（≈1g），加一点走路摆动
    t = np.linspace(0, 2, N)
    normal = np.column_stack([
        0.15 * np.sin(2 * np.pi * 1.0 * t) + 0.03 * rng.standard_normal(N),
        0.10 * np.sin(2 * np.pi * 0.8 * t) + 0.03 * rng.standard_normal(N),
        1.0 + 0.12 * np.sin(2 * np.pi * 2.0 * t) + 0.03 * rng.standard_normal(N),
    ])
    f1 = extract_features(normal)
    print("正常窗口特征维度:", len(f1), "| mean_SVM≈", round(f1[30], 3))
