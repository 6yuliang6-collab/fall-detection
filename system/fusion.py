"""
视觉 + IMU 决策级融合（多模态）。

两个模态各自独立推理，各给一个 (类别, 置信度)，这里把它们合并成最终报警决策。

输入：
  vision: dict, 例如 {"cls": 1, "conf": 0.93}   cls: 0=正常 1=跌倒
  imu:    dict, 例如 {"cls": 1, "prob": 0.99}   prob 为 IMU 模型输出的跌倒概率 [0,1]

规则：
  "or"      : 任一判跌倒就报警（漏报最少、误报最多）
  "and"     : 两者都判跌倒才报警（误报最少、漏报最多）
  "weighted": score = w*视觉 + (1-w)*IMU，>= threshold 才报警（折中，推荐）

用法：
  from fusion import fuse
  alert, score = fuse(vision={"cls":1,"conf":0.9}, imu={"cls":0,"prob":0.3}, rule="weighted")
"""


def _vision_prob(vision):
    """视觉置信度转成「是跌倒」的概率。"""
    c = vision.get("conf", 0.0)
    return c if vision.get("cls") == 1 else (1 - c)


def fuse(vision, imu, rule="weighted", w=0.5, threshold=0.5):
    """
    返回 (是否报警 bool, 融合分数 float)。
    w 是视觉的权重（weighted 规则用），跌倒场景建议视觉稍低权重（因为 IMU 更准），
    例如 w=0.4 表示 IMU 占 0.6。
    """
    vp = _vision_prob(vision)
    ip = float(imu.get("prob", 0.0))
    if imu.get("cls") == 0:
        ip = 1 - ip  # IMU 判非跌倒，取「是跌倒」概率 = 1 - prob(非跌倒)

    if rule == "or":
        alert = (vision.get("cls") == 1) or (imu.get("cls") == 1)
        return alert, max(vp, ip)
    if rule == "and":
        alert = (vision.get("cls") == 1) and (imu.get("cls") == 1)
        return alert, min(vp, ip)
    # weighted
    score = w * vp + (1 - w) * ip
    return score >= threshold, score


if __name__ == "__main__":
    # 几个示例
    cases = [
        ("都判跌倒", {"cls": 1, "conf": 0.93}, {"cls": 1, "prob": 0.99}),
        ("视觉判跌倒/IMU判正常", {"cls": 1, "conf": 0.80}, {"cls": 0, "prob": 0.60}),
        ("视觉漏检(遮挡)/IMU判跌倒", {"cls": 0, "conf": 0.55}, {"cls": 1, "prob": 0.97}),
        ("都判正常", {"cls": 0, "conf": 0.90}, {"cls": 0, "prob": 0.95}),
    ]
    print(f"{'场景':<22}{'OR':>8}{'AND':>8}{'加权':>8}")
    for name, v, i in cases:
        o, _ = fuse(v, i, rule="or")
        a, _ = fuse(v, i, rule="and")
        w_, s = fuse(v, i, rule="weighted", w=0.4)
        print(f"{name:<22}{str(o):>8}{str(a):>8}{str(w_) + f'({s:.2f})':>8}")
