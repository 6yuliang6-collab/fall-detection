"""
模拟 IMU 边缘端：发布 IMU 跌倒概率到 MQTT（无 MPU6050/ESP32 硬件时的演示用）。

用法（项目根目录）：
  # 模拟持续跌倒（配视觉一起跑，演示融合报警）
  venv\\Scripts\\python.exe system\\imu_simulator.py --broker 8.219.124.194 --mode fall
  # 模拟正常活动
  venv\\Scripts\\python.exe system\\imu_simulator.py --broker 8.219.124.194 --mode normal
  # 每 5 秒在"跌倒/正常"间切换（演示融合的实时响应）
  venv\\Scripts\\python.exe system\\imu_simulator.py --broker 8.219.124.194 --mode alternate
"""
import argparse
import json
import time

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="fall/imu/events")
    ap.add_argument("--mode", choices=["fall", "normal", "alternate"], default="fall")
    ap.add_argument("--device", default="imu01")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    c = mqtt.Client()
    c.connect(args.broker, args.port, keepalive=60)
    c.loop_start()
    print(f"[IMU模拟] 已连接 {args.broker}:{args.port}，模式={args.mode}")

    i = 0
    try:
        while True:
            if args.mode == "fall":
                cls, prob = "down", 0.95
            elif args.mode == "normal":
                cls, prob = "normal", 0.95
            else:  # alternate：每 10 条(约5秒)切换
                cls, prob = ("down", 0.95) if (i // 10) % 2 == 0 else ("normal", 0.95)
            c.publish(args.topic, json.dumps({"device": args.device, "class_name": cls, "prob": prob}))
            print(f"[IMU模拟] 发布: {cls} (prob={prob})")
            time.sleep(args.interval)
            i += 1
    except KeyboardInterrupt:
        pass
    c.disconnect()
    print("[IMU模拟] 已退出")


if __name__ == "__main__":
    main()
