"""
云端后端：MQTT 订阅 + 事件存储 + 警报规则 + Web 仪表盘

职责（模拟部署在云端）：
  1. 订阅边缘端发来的检测事件（MQTT）
  2. 事件落库（SQLite）
  3. 应用警报规则（连续 N 帧 down 才告警，去抖防误报）
  4. 提供 REST API + 内置 Web 仪表盘

用法（在项目根目录执行）：
  venv\\Scripts\\python.exe system\\cloud_backend.py
  然后浏览器打开 http://127.0.0.1:5000
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from flask import Flask, jsonify, send_from_directory, request

import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# ---------- SQLite ----------
_DB_LOCK = threading.Lock()


def db():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _DB_LOCK:
        c = db()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                modality TEXT,
                device TEXT,
                class_name TEXT,
                conf REAL,
                box TEXT
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                device TEXT,
                reason TEXT
            );
        """)
        c.commit()
        c.close()


# ---------- 多模态融合状态 + 警报规则 ----------
_CLS = {"down": 1, "normal": 0, "none": 0}

_fusion_state = {
    "vision": {"cls": 0, "conf": 0.0},   # 最新视觉结果
    "imu": {"cls": 0, "prob": 0.0},      # 最新 IMU 结果
    "fused_down": 0,                     # 融合后连续判 down 计数
    "last_alert_ts": 0.0,                # 上次告警时间（冷却）
}


# 纯视觉模式的去抖状态（FUSION_ENABLED=False 时用）
_vision_state = {"consecutive_down": 0, "last_alert_ts": 0.0}


def apply_vision_alert(device, class_name, conf):
    """纯视觉去抖警报规则：连续 N 帧 down 才报警。"""
    with _DB_LOCK:
        if class_name == "down" and conf >= config.DOWN_CONF_THRESHOLD:
            _vision_state["consecutive_down"] += 1
        elif class_name == "normal":
            _vision_state["consecutive_down"] = 0

        if _vision_state["consecutive_down"] >= config.ALERT_CONSECUTIVE_DOWN:
            now = time.time()
            if now - _vision_state["last_alert_ts"] >= config.ALERT_COOLDOWN_SEC:
                _vision_state["last_alert_ts"] = now
                _vision_state["consecutive_down"] = 0
                return True
        return False


def apply_fusion_alert(device):
    """视觉 + IMU 决策级融合，返回 (是否报警, 融合分数)。"""
    import fusion
    v = _fusion_state["vision"]
    i = _fusion_state["imu"]
    alerted, score = fusion.fuse(v, i, rule=config.FUSION_RULE,
                                 w=config.FUSION_VISION_WEIGHT, threshold=0.5)
    with _DB_LOCK:
        if alerted:
            _fusion_state["fused_down"] += 1
        else:
            _fusion_state["fused_down"] = 0

        if _fusion_state["fused_down"] >= config.ALERT_CONSECUTIVE_DOWN:
            now = time.time()
            if now - _fusion_state["last_alert_ts"] >= config.ALERT_COOLDOWN_SEC:
                _fusion_state["last_alert_ts"] = now
                _fusion_state["fused_down"] = 0
                return True, score
        return False, score


def insert_event(modality, device, class_name, conf, box):
    with _DB_LOCK:
        c = db()
        c.execute("INSERT INTO events (ts, modality, device, class_name, conf, box) VALUES (?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), modality, device, class_name, conf, json.dumps(box)))
        c.commit()
        c.close()


def insert_alert(device, reason):
    with _DB_LOCK:
        c = db()
        c.execute("INSERT INTO alerts (ts, device, reason) VALUES (?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), device, reason))
        c.commit()
        c.close()


# ---------- MQTT 订阅 ----------
def on_connect(client, userdata, flags, rc):
    client.subscribe(config.MQTT_TOPIC_EVENTS)
    client.subscribe(config.MQTT_TOPIC_IMU)
    print(f"[云端] 已订阅: {config.MQTT_TOPIC_EVENTS} (视觉) + {config.MQTT_TOPIC_IMU} (IMU)")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print("[云端] 消息解析失败:", e)
        return

    modality = "vision" if msg.topic == config.MQTT_TOPIC_EVENTS else "imu"
    device = data.get("device", "unknown")
    class_name = data.get("class_name", "none")
    conf = float(data.get("conf", data.get("prob", 0.0)))
    dets = data.get("detections", [])

    # 存储事件（带模态）
    insert_event(modality, device, class_name, conf,
                 dets[0].get("box") if dets else [])

    # 报警判定：纯视觉 或 多模态融合（由 config.FUSION_ENABLED 控制）
    if config.FUSION_ENABLED:
        # 更新该模态的最新结果，供融合使用
        cls_int = _CLS.get(class_name, 0)
        if modality == "vision":
            _fusion_state["vision"] = {"cls": cls_int, "conf": conf}
        else:
            _fusion_state["imu"] = {"cls": cls_int, "prob": conf}

        alerted, score = apply_fusion_alert(device)
        if alerted:
            insert_alert(device, f"多模态融合判定跌倒 (融合分数 {score:.2f})")
            print(f"[云端] [警报] 融合判定跌倒！设备 {device} score={score:.2f}")
    else:
        # 纯视觉判定
        if apply_vision_alert(device, class_name, conf):
            insert_alert(device, f"视觉连续 {config.ALERT_CONSECUTIVE_DOWN} 帧判为跌倒")
            print(f"[云端] [警报] 触发警报！设备 {device} 检测到跌倒")


# ---------- REST API ----------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/stats")
def api_stats():
    with _DB_LOCK:
        c = db()
        total = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        down = c.execute("SELECT COUNT(*) FROM events WHERE class_name='down'").fetchone()[0]
        normal = c.execute("SELECT COUNT(*) FROM events WHERE class_name='normal'").fetchone()[0]
        alerts = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        c.close()
    return jsonify({
        "total_events": total,
        "down": down,
        "normal": normal,
        "alerts": alerts,
    })


@app.route("/api/latest")
def api_latest():
    with _DB_LOCK:
        c = db()
        row = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT 1").fetchone()
        c.close()
    return jsonify(dict(row) if row else {})


@app.route("/api/events")
def api_events():
    limit = min(int(request.args.get("limit", 30)), 200)
    with _DB_LOCK:
        c = db()
        rows = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        c.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/alerts")
def api_alerts():
    limit = min(int(request.args.get("limit", 30)), 200)
    with _DB_LOCK:
        c = db()
        rows = c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        c.close()
    return jsonify([dict(r) for r in rows])


def start_mqtt():
    client = mqtt.Client(client_id=f"cloud-{os.getpid()}", protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=60)
        client.loop_start()
        print(f"[云端] MQTT 已连接 {config.BROKER_HOST}:{config.BROKER_PORT}")
    except Exception as e:
        print(f"[云端] [!] MQTT broker 连接失败（{e}），仪表盘仍可访问，但收不到边缘事件")


if __name__ == "__main__":
    init_db()
    start_mqtt()
    print(f"[云端] 仪表盘已启动: http://{config.WEB_HOST}:{config.WEB_PORT}")
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False, use_reloader=False)
