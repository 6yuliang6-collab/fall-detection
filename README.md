# 跌倒检测系统（视觉 + IMU 双模态 + 云端后端）

基于 YOLOv5（视觉）和 IMU（随机森林/1D-CNN）的多模态跌倒检测系统，通过 MQTT 把边缘端检测结果发送到**云服务器**，云端做融合判定、存储、警报，并通过 Web 仪表盘展示。

## 系统架构（云服务器参与过程）

```
┌─────────────────────────────────────────────────────────────────────┐
│  边缘端（你电脑 /                                           │
│                                                                     │
│  [视觉] 摄像头 ──► YOLOv5(fall_v5) ──► 检测 normal/down             │
│                                            │                        │
│  [IMU]  MPU6050 ──► 随机森林/1D-CNN ──► 跌倒概率                     │
│                                            │                        │
└────────────────────────────────────────────┼────────────────────────┘
                                             │ MQTT 发布
                                             ▼
                    ┌────────────────────────────────┐
                    │      云服务器 (阿里云 8.219.124.194) │
                    │                                │
                    │  mosquitto (MQTT broker :1883)  │ ◄── 接收边缘数据
                    │        │                       │
                    │  Flask 后端 (:5000)            │ ◄── 订阅 MQTT
                    │    ├─ 多模态融合 / 纯视觉判定   │
                    │    ├─ SQLite 数据库 (存事件/警报)│
                    │    ├─ 去抖警报规则              │
                    │    └─ Web 仪表盘                │
                    └───────────────┬────────────────┘
                                    │ HTTP
                                    ▼
                     浏览器 http://8.219.124.194:5000  ◄── 用户查看
```

**云服务器参与的关键点**：边缘端只做检测，**不存储、不判警报**；所有事件通过 MQTT 发到云端，由云端完成存储、融合判定、去抖报警，用户通过公网 URL 访问仪表盘查看结果。

## 环境依赖

完整环境（训练 + 推理）见 [requirements.txt](requirements.txt)。核心：
- Python 3.11
- torch 2.14.0+cu126、opencv-python 5.0
- flask、paho-mqtt、scikit-learn

云服务器端只需要（见 [deploy/requirements.txt](deploy/requirements.txt)）：
- flask、paho-mqtt（**不需要 torch/GPU**，模型推理在边缘端）

## 可重现步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 复现训练（视觉模型）
```bash
# 数据集：Roboflow fall-awbxa（下载见 tools/fetch_roboflow.py）+ 自采摄像头数据
# 切分 + 合并：tools/build_final_dataset.py、tools/merge_capture.py
python yolov5/train.py --img 640 --batch 16 --epochs 150 --data data_final.yaml \
  --weights yolov5s.pt --device 0 --hyp yolov5/data/hyps/hyp.fall3.yaml --cos-lr
```

### 3. 复现 IMU 模型（RF vs 1D-CNN）
```bash
python tools/imu_rf_cnn.py   # 训练并输出 P/R/F1
python tools/imu_viz.py      # 生成对比图
```

### 4. 本地运行（边缘 + 云端都在本机）
```bash
# 终端1：云端后端
python system/cloud_backend.py
# 终端2：摄像头检测
python system/edge_detector.py --source 0
# 浏览器打开 http://127.0.0.1:5000
```

### 5. 云服务器部署（见 deploy/README.md 完整步骤）
```bash
# 上传代码 + 跑部署脚本
scp -r system deploy admin@<公网IP>:/home/admin/fall-detection/
ssh admin@<公网IP> "bash /home/admin/fall-detection/deploy/setup.sh"
```

## 演示工作流（1-2 分钟演示脚本）

1. **展示云服务器已部署**：浏览器打开 `http://8.219.124.194:5000`（公网仪表盘）
2. **启动边缘检测**：`python system/edge_detector.py --source 0 --broker 8.219.124.194`
3. **对着摄像头做动作**：站立 → 坐下/躺下 → 站立
4. 仪表盘实时刷新事件，连续跌倒触发警报
5. **证明是云端**：`curl http://8.219.124.194:5000/api/stats` 返回 JSON；服务器日志显示请求；数据库有记录

## 评估结果（保留测试集）

| 模态 / 模型 | 类别 | P | R | F1 | mAP50 |
|---|---|---|---|---|---|
| 视觉 YOLOv5 (fall_v5) | down | 0.966 | 0.956 | 0.961 | 0.989 |
| IMU 随机森林 | fall | 0.994 | 0.993 | 0.994 | — |
| IMU 1D-CNN | fall | 0.977 | 0.972 | 0.975 | — |

## 目录结构

```
system/        边缘检测(edge_detector.py) + 云端后端(cloud_backend.py) + 融合(fusion.py) + 仪表盘
deploy/        云服务器部署脚本 + 文档
tools/         数据下载、切分、标注、误报分析、IMU 模型训练脚本
yolov5/        YOLOv5 框架 + 训练
dataset_final/ 最终训练数据（Roboflow + 自采）
runs/train/    训练结果（fall_v5 等）
```

## 数据与模型来源（学术诚信引用）

- 视觉数据：Roboflow [fall-awbxa](https://universe.roboflow.com/go-ygbsj/fall-awbxa)（CC BY 4.0）+ 自采摄像头数据
- IMU 数据：SisFall（Sucerquia et al., 2017）
- 基线模型：YOLOv5（Ultralytics, AGPL-3.0）、scikit-learn RandomForest
