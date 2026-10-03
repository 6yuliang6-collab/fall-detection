# 跌倒检测系统（MQTT → 云端 → 仪表盘 → 警报）

把训练好的 YOLOv5 模型接到一条端到端系统链路上：边缘检测 → MQTT → 云端存储/警报 → Web 仪表盘。

## 架构

```
┌───────────────┐  MQTT   ┌─────────────┐  HTTP   ┌──────────────┐
│ 边缘端(相机)   │ ──────▶ │  MQTT Broker │ ──────▶ │  云端后端     │──▶ Web 仪表盘
│ edge_detector │ 发布     │ (mosquitto) │ 订阅     │ cloud_backend │    (index.html)
└───────────────┘         └─────────────┘         └──────────────┘
                                                        │ SQLite 存储
                                                        │ 警报规则(去抖)
```

## 依赖

```bash
venv\Scripts\python.exe -m pip install paho-mqtt flask
```

## MQTT Broker（二选一）

**方式 A：本地 mosquitto（推荐，离线可用）**

```bash
# Windows 用 winget 安装：
winget install EclipseFoundation.Mosquitto
# 装完手动启动：
mosquitto -p 1883
```

**方式 B：公共 broker（零安装，需联网）**

把 [config.py](config.py) 里的 `BROKER_HOST = "localhost"` 改成 `BROKER_HOST = "broker.emqx.io"`。

## 运行（开两个终端）

**终端 1 —— 云端后端（先启动）：**

```bash
cd /d C:\Users\Lenovo\fall-detection
venv\Scripts\python.exe system\cloud_backend.py
```

浏览器打开 http://127.0.0.1:5000 看仪表盘。

**终端 2 —— 边缘检测（发布事件）：**

```bash
cd /d C:\Users\Lenovo\fall-detection
:: 用摄像头
venv\Scripts\python.exe system\edge_detector.py --source 0
:: 或用一个视频文件
venv\Scripts\python.exe system\edge_detector.py --source 你的视频.mp4
:: 或用验证集图片目录批量演示
venv\Scripts\python.exe system\edge_detector.py --source dataset\images\val
```

## 警报规则（防误报的核心）

在 [config.py](config.py) 里可调：

- `ALERT_CONSECUTIVE_DOWN = 3`：**连续 3 帧判为 down 才告警**。单帧误判（比如正常动作被瞬间误判成跌倒）不会触发警报，这是典型的"时间去抖"。
- `ALERT_COOLDOWN_SEC = 10`：同一次跌倒只告警一次，冷却期内不重复刷屏。
- `DOWN_CONF_THRESHOLD = 0.5`：置信度阈值。

这三个参数直接决定"误报 vs 漏报"的权衡：阈值/去抖越高 → 误报越少、漏报越多。

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | 所有可调参数集中在这 |
| `edge_detector.py` | 边缘端：检测 + MQTT 发布 |
| `cloud_backend.py` | 云端：MQTT 订阅 + SQLite + 警报 + REST API + 仪表盘 |
| `static/index.html` | Web 仪表盘 |
| `fall_events.db` | 运行时自动生成的 SQLite 数据库 |
