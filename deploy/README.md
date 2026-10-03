# 阿里云部署指南（把后端 + MQTT 部署到云服务器）

目标：把 `cloud_backend.py`（Flask + SQLite + 仪表盘）+ mosquitto 部署到阿里云 ECS，
让公网能访问仪表盘，边缘端（你的电脑）通过公网 IP 连 MQTT。

架构：

```
你的电脑(边缘) ──检测──► MQTT ──► 阿里云服务器 ──► Flask 后端 + 仪表盘
   edge_detector.py         (公网IP:1883)    mosquitto + cloud_backend.py
                                              (公网IP:5000)
```

## 第 1 步：创建/确认 ECS 实例

1. 登录阿里云控制台 → 云服务器 ECS → 创建实例
2. 镜像选 **Ubuntu 20.04 或 22.04**（推荐，脚本按 Ubuntu 写的）
3. 最低配 1 核 1G 就够（只跑 Flask + mosquitto，很轻）
4. 记下**公网 IP**

## 第 2 步：配置安全组（放行端口，关键！）

ECS 控制台 → 安全组 → 配置规则 → 添加入方向规则：

| 端口 | 用途 |
|---|---|
| 22 | SSH 登录 |
| 5000 | Flask 仪表盘 |
| 1883 | MQTT |

授权对象填 `0.0.0.0/0`（演示期可以全放行；正式环境建议限制 IP）。

## 第 3 步：上传代码到服务器

只需要这三个东西（都是纯 Python，服务器上**不用装 torch/GPU**）：
- `system/cloud_backend.py`
- `system/config.py`
- `system/static/`（整个文件夹）

**方式 A：scp（Windows 用 Git Bash / PowerShell）**
```bash
# 在项目根目录 C:\Users\Lenovo\fall-detection 执行
scp -r system root@<公网IP>:/root/fall-detection/
scp -r deploy root@<公网IP>:/root/fall-detection/
```

**方式 B：WinSCP 图形界面**（拖拽上传更方便）

## 第 4 步：SSH 登录并运行部署脚本

```bash
ssh root@<公网IP>
# 登录后执行：
bash ~/fall-detection/deploy/setup.sh
```

脚本会自动：装 mosquitto + python 依赖 → 配置 mosquitto 监听公网 → 后台启动 Flask。

## 第 5 步：验证云后端在运行（这就是"云运行证据"）

```bash
# 服务器上
curl http://127.0.0.1:5000/api/stats      # 应返回 {"total_events":0,...}

# 你电脑浏览器打开（这就是老师要的证据）
http://<公网IP>:5000
```

## 第 6 步：让边缘端（你的电脑）连云服务器

你电脑上的 `edge_detector.py` 现在把检测结果发到**云服务器的 MQTT**：

```bash
cd /d C:\Users\Lenovo\fall-detection
venv\Scripts\python.exe system\edge_detector.py --source 0 --broker <公网IP>
```

云端仪表盘（`http://<公网IP>:5000`）就会实时显示你摄像头检测到的跌倒事件和警报。

## 常见问题

- **浏览器打不开仪表盘** → 检查安全组 5000 端口是否放行、后端是否在跑（`ps aux | grep cloud_backend`）
- **边缘端连不上 MQTT** → 检查安全组 1883 端口、mosquitto 是否监听 0.0.0.0（`ss -tlnp | grep 1883`）
- **看后端日志** → `cat ~/fall-detection/cloud.log`
- **重启后端** → `pkill -f cloud_backend && cd ~/fall-detection && nohup python3 system/cloud_backend.py > cloud.log 2>&1 &`
