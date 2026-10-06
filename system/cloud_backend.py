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
        # 迁移：兼容旧库（旧库的 events 表没有 modality 列时补上）
        cols = [r[1] for r in c.execute("PRAGMA table_info(events)").fetchall()]
        if "modality" not in cols:
            c.execute("ALTER TABLE events ADD COLUMN modality TEXT")
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

# 融合模式（运行时可通过 /api/fusion/mode 切换，无需重启）
_fusion_config = {
    "rule": config.FUSION_RULE,          # "weighted" / "or" / "and"
    "w": config.FUSION_VISION_WEIGHT,    # 视觉权重（weighted 规则用）
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
    alerted, score = fusion.fuse(v, i, rule=_fusion_config["rule"],
                                 w=_fusion_config["w"], threshold=0.5)
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
    if config.IMU_INFERENCE_ON_CLOUD:
        client.subscribe(config.MQTT_TOPIC_IMU_RAW)
        print(f"[云端] 已订阅: {config.MQTT_TOPIC_EVENTS} (视觉) + "
              f"{config.MQTT_TOPIC_IMU_RAW} (IMU原始窗口, 云端RF推理)")
    else:
        client.subscribe(config.MQTT_TOPIC_IMU)
        print(f"[云端] 已订阅: {config.MQTT_TOPIC_EVENTS} (视觉) + "
              f"{config.MQTT_TOPIC_IMU} (IMU)")


def run_imu_inference(device, window):
    """云端 IMU 推理：原始窗口 -> 特征 -> 随机森林 -> (class_name, conf)。"""
    from imu_rf_inference import predict
    cls, p_fall = predict(window)
    class_name = "down" if cls == 1 else "normal"
    conf = p_fall if cls == 1 else (1.0 - p_fall)  # 预测类别的置信度
    print(f"[云端] [IMU推理] 设备 {device} -> {class_name} (跌倒概率 {p_fall:.4f})")
    return class_name, conf


def handle_event(modality, device, class_name, conf, dets):
    """统一处理一个事件：落库 + 报警判定（纯视觉 或 多模态融合）。"""
    insert_event(modality, device, class_name, conf,
                 dets[0].get("box") if dets else [])

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


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        print("[云端] 消息解析失败:", e)
        return

    # ---- 主题路由 ----
    if msg.topic == config.MQTT_TOPIC_EVENTS:
        # 视觉：边缘端已检测好，直接带 class_name/conf
        modality = "vision"
        device = data.get("device", "unknown")
        class_name = data.get("class_name", "none")
        conf = float(data.get("conf", 0.0))
        dets = data.get("detections", [])
    elif msg.topic == config.MQTT_TOPIC_IMU_RAW:
        # IMU 原始加速度窗口：云端提取特征 + 跑随机森林推理
        try:
            device = data.get("device", "unknown")
            class_name, conf = run_imu_inference(device, data.get("window", []))
        except Exception as e:
            print(f"[云端] [IMU推理] 推理失败: {e}")
            return
        modality = "imu"
        dets = []
    else:
        # IMU 边缘端已判好的结论（IMU_INFERENCE_ON_CLOUD=False 时）
        modality = "imu"
        device = data.get("device", "unknown")
        class_name = data.get("class_name", "none")
        conf = float(data.get("conf", data.get("prob", 0.0)))
        dets = []

    handle_event(modality, device, class_name, conf, dets)


# ---------- REST API ----------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/phone_imu")
def phone_imu():
    return send_from_directory(STATIC_DIR, "phone_imu.html")


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


@app.route("/api/imu/raw", methods=["POST"])
def api_imu_raw():
    """手机/HTTP 客户端直接 POST 原始加速度窗口，云端跑 RF 推理。
    请求体: {"device":"phone01", "fs":50, "window":[[ax,ay,az], ...]}
    """
    data = request.get_json(silent=True) or {}
    device = data.get("device", "phone01")
    window = data.get("window", [])
    if not window:
        return jsonify({"error": "missing window"}), 400
    try:
        class_name, conf = run_imu_inference(device, window)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500
    handle_event("imu", device, class_name, conf, [])
    return jsonify({"device": device, "class_name": class_name, "conf": round(conf, 4)})


@app.route("/api/fusion/mode", methods=["GET", "POST"])
def api_fusion_mode():
    """查询或切换融合模式（演示时无需重启）。
    POST 体：{"rule": "weighted"|"or"|"and", "w": 0.6}
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if data.get("rule") in ("weighted", "or", "and"):
            _fusion_config["rule"] = data["rule"]
        if "w" in data:
            try:
                w = float(data["w"])
                if 0.0 <= w <= 1.0:
                    _fusion_config["w"] = w
            except (TypeError, ValueError):
                pass
        print(f"[云端] 融合模式切换为: rule={_fusion_config['rule']}, "
              f"视觉权重={_fusion_config['w']}")
    return jsonify({"rule": _fusion_config["rule"],
                    "vision_weight": _fusion_config["w"]})


@app.after_request
def add_cors(resp):
    # 允许手机网页从任意来源调用（同一域名其实不需要，加了更稳）
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


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


_INITIALIZED = False


def ensure_init():
    """建库 + 启动 MQTT（幂等）。gunicorn 导入模块时也会触发一次。"""
    global _INITIALIZED
    if _INITIALIZED:
        return
    init_db()
    start_mqtt()
    _INITIALIZED = True


# 模块导入即初始化（gunicorn `cloud_backend:app` 依赖这行）
ensure_init()


if __name__ == "__main__":
    # 直接运行（开发模式）；生产用 gunicorn，见 deploy/README.md
    ensure_init()
    ssl_context = None
    scheme = "http"
    if config.SSL_ENABLED and os.path.exists(config.SSL_CERT) and os.path.exists(config.SSL_KEY):
        ssl_context = (config.SSL_CERT, config.SSL_KEY)
        scheme = "https"
    print(f"[云端] 仪表盘已启动: {scheme}://{config.WEB_HOST}:{config.WEB_PORT}")
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, ssl_context=ssl_context,
            debug=False, use_reloader=False, threaded=True)
