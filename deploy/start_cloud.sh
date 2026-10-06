#!/bin/bash
# 生产级启动：gunicorn 多线程 + HTTPS（自签名证书）
# 用法：bash ~/fall-detection/deploy/start_cloud.sh
export PATH="$PATH:$HOME/.local/bin"
cd "$(dirname "$0")/../system" || exit 1
exec gunicorn \
    --workers=1 \
    --threads=8 \
    --certfile=cert.pem \
    --keyfile=key.pem \
    -b 0.0.0.0:5000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    cloud_backend:app
