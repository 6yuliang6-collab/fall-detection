# 项目摘要（Abstract）

We use datasets such as **SisFall (IMU accelerometer features, 9,000 samples) and a Roboflow
fall-detection dataset (vision, 5,244 images)**, and extend them by **comparing Random Forest
with a 1D-CNN on the IMU modality, and training a YOLOv5 detector on the vision modality
(further fine-tuned on self-collected camera data)**. We also add **MQTT streaming, a Flask
cloud API, a SQLite database, a live web dashboard, and a debounced alert rule**. The cloud
backend is deployed on **a local machine (edge and cloud on a single Windows host)** and runs
**Mosquitto (MQTT broker), Flask (API + dashboard), SQLite, and paho-mqtt**. Experimental
results on a held-out test set show that **Random Forest achieves 0.99 F1 with 0.99 precision
and 0.99 recall, while YOLOv5 achieves 0.96 F1 (P=0.97, R=0.96) on the fall class**. The
end-to-end latency is **about 10 ms per frame (4.8 ms GPU inference)**, with a 1.5 s debounce
window before an alert is triggered. The project targets the Intermediate technical level.

---

## 关键指标速查

| 模态 / 模型 | 类别 | P | R | F1 | mAP50 |
|---|---|---|---|---|---|
| 视觉 YOLOv5 (fall_v5) | down(跌倒) | 0.966 | 0.956 | 0.961 | 0.989 |
| 视觉 YOLOv5 (fall_v5) | all | 0.953 | 0.955 | 0.954 | 0.977 |
| IMU 随机森林 RF | fall(跌倒) | 0.994 | 0.993 | 0.994 | — |
| IMU 1D-CNN | fall(跌倒) | 0.977 | 0.972 | 0.975 | — |

数据来源：
- 视觉：Roboflow [fall-awbxa](https://universe.roboflow.com/go-ygbsj/fall-awbxa)（CC BY 4.0）+ 自采摄像头数据微调
- IMU：SisFall（Sucerquia et al., 2017）
