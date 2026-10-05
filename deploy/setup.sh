#!/bin/bash
# 阿里云服务器一键部署脚本（Ubuntu 20.04/22.04）
# 用法：先按 README.md 把代码传到服务器 ~/fall-detection，然后执行：
#   bash ~/fall-detection/deploy/setup.sh
set -e

echo "=== [1/4] 安装 mosquitto + python3 ==="
sudo apt-get update -y
sudo apt-get install -y mosquitto mosquitto-clients python3 python3-pip

echo "=== [2/4] 安装 Python 依赖 ==="
pip3 install --user -r ~/fall-detection/deploy/requirements.txt

echo "=== [3/4] 配置 mosquitto 监听所有网卡 + 允许匿名 ==="
sudo tee /etc/mosquitto/conf.d/fall.conf > /dev/null <<'EOF'
listener 1883 0.0.0.0
allow_anonymous true
EOF
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto

echo "=== [4/4] 后台启动 Flask 后端 ==="
cd ~/fall-detection
nohup python3 system/cloud_backend.py > cloud.log 2>&1 &
sleep 2
echo ""
echo "部署完成！"
echo "  - 仪表盘:   http://<服务器公网IP>:5000"
echo "  - MQTT:     <服务器公网IP>:1883"
echo "  - 后端日志: ~/fall-detection/cloud.log"
echo ""
echo "验证： curl http://127.0.0.1:5000/api/stats"
