"""
系统共享配置（边缘检测 + 云端后端 + 仪表盘）

所有可调参数都集中在这里，改这一处即可。
"""
import os

# ---------- 项目路径 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fall-detection/
YOLOV5_DIR = os.path.join(PROJECT_ROOT, "yolov5")
# 训练好的模型（fall_v5 是加了摄像头采集数据微调版：test 集 down F1 0.961、mAP50-95 0.824）
DEFAULT_WEIGHTS = os.path.join(YOLOV5_DIR, "runs", "train", "fall_v5", "weights", "best.pt")

# ---------- MQTT ----------
# 本地 broker（推荐装 mosquitto，见 README.md）
# 想零安装立刻演示：改成 "broker.emqx.io"（公共免费 broker，需联网）
BROKER_HOST = "localhost"
BROKER_PORT = 1883
# 主题：边缘端把检测结果发到这个主题，云端订阅
MQTT_TOPIC_EVENTS = "fall/detection/events"   # 视觉模态
MQTT_TOPIC_IMU = "fall/imu/events"            # IMU 模态

# ---------- 多模态融合 ----------
FUSION_ENABLED = False        # True=视觉+IMU 融合判定；False=纯视觉判定
FUSION_RULE = "weighted"      # or / and / weighted
FUSION_VISION_WEIGHT = 0.4    # 加权融合里视觉的权重（IMU 占 0.6，因为 IMU 更准）

# ---------- 云端后端 ----------
WEB_HOST = "0.0.0.0"   # 绑定所有网卡：本地 http://127.0.0.1:5000，云上 http://<公网IP>:5000 都能访问
WEB_PORT = 5000
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fall_events.db")

# ---------- 警报规则（核心：去抖防误报）----------
# 连续 N 帧判为 down（中间没有 normal 打断）才触发警报
# 值越大越能压误报、但漏报风险上升；值越小越灵敏、但误报多
ALERT_CONSECUTIVE_DOWN = 3
# 判为目标的置信度阈值（边缘端和云端共用）
# fall_v4 用标准超参（无 focal loss），置信度校准正常。
# 实际摄像头场景比 Roboflow 训练集有偏移，杂物易误报，已从 0.25 抬到 0.35 压误报。
# 若发现真人漏检，可再降回 0.25~0.30。
DOWN_CONF_THRESHOLD = 0.35
# 一条警报触发后的冷却时间（秒），冷却期内同一设备不再重复告警
ALERT_COOLDOWN_SEC = 10
