/*
 * IMU 边缘端：ESP32 + MPU6050 -> WiFi -> MQTT（云端推理版）
 *
 * 读原始加速度，攒 2 秒窗口（100 采样 @50Hz），发 JSON 到 MQTT，
 * 由云端提取特征 + 跑随机森林推理（不要再在 ESP32 上做启发式判断）。
 *
 * 依赖库（Arduino IDE 库管理器搜索安装）：
 *   - MPU6050_tockn   (作者 tockn)
 *   - PubSubClient    (作者 Nick O'Leary)
 *   （WiFi.h 是 ESP32 自带，无需额外装）
 *
 * 接线：
 *   MPU6050  VCC -> 3.3V   GND -> GND   SDA -> GPIO21   SCL -> GPIO22
 *
 * 消息格式（和 system/imu_raw_simulator.py 完全一致，云端无缝接收）：
 *   {"device":"imu01","fs":50,"window":[[ax,ay,az],[ax,ay,az], ... 共100个]}
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <MPU6050_tockn.h>

// ============ 改成你自己的 ============
const char* WIFI_SSID   = "你的WiFi名";
const char* WIFI_PASS   = "你的WiFi密码";
// 云端公网 broker（演示直接用云）；也可以改成你电脑局域网 IP 连本地 mosquitto
const char* MQTT_BROKER = "8.219.124.194";
const int   MQTT_PORT   = 1883;
const char* TOPIC       = "fall/imu/raw";   // 关键：原始窗口 topic（云端 RF 推理）
const char* DEVICE      = "imu01";
// ======================================

const int FS    = 50;             // 采样率 Hz
const int WIN_S = 2;              // 窗口秒
const int NSAMP = FS * WIN_S;     // 100 个采样

WiFiClient   espClient;
PubSubClient client(espClient);
MPU6050      mpu(Wire);

float window[NSAMP][3];           // 100 × (ax, ay, az)，单位 g
int   wcount = 0;
unsigned long lastSample = 0;

void reconnect() {
  while (!client.connected()) {
    Serial.print("连接 MQTT...");
    if (client.connect("esp32-imu")) {
      Serial.println("成功");
      client.subscribe(TOPIC);     // 可选，仅用于回显确认
    } else {
      Serial.print("失败, rc=");
      Serial.print(client.state());
      Serial.println(" 5秒后重试");
      delay(5000);
    }
  }
}

void publishWindow() {
  // 手动拼 JSON（避免额外依赖 ArduinoJson）。
  // 100 个采样约 2KB，默认 128 字节缓冲区放不下，所以 setup 里 setBufferSize(4096)。
  static char payload[4096];
  int pos = 0;
  pos += snprintf(payload + pos, sizeof(payload) - pos,
                  "{\"device\":\"%s\",\"fs\":%d,\"window\":[", DEVICE, FS);
  for (int i = 0; i < NSAMP; i++) {
    pos += snprintf(payload + pos, sizeof(payload) - pos,
                    "[%.3f,%.3f,%.3f]%s",
                    window[i][0], window[i][1], window[i][2],
                    (i == NSAMP - 1) ? "" : ",");
  }
  pos += snprintf(payload + pos, sizeof(payload) - pos, "]}");

  client.publish(TOPIC, payload);
  Serial.printf("已发布窗口 %d 采样, %d 字节\n", NSAMP, pos);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.begin();
  mpu.calcGyroOffsets(true);   // 校准，此时保持传感器静止

  Serial.print("连接 WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(500); }
  Serial.println();
  Serial.print("WiFi 已连接, IP: ");
  Serial.println(WiFi.localIP());

  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setBufferSize(4096);   // 关键：发 2KB 窗口必须加大缓冲区
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  // 50Hz 采样（每 20ms 一次）
  if (millis() - lastSample >= 20) {
    lastSample = millis();
    mpu.update();
    window[wcount][0] = mpu.getAccX();
    window[wcount][1] = mpu.getAccY();
    window[wcount][2] = mpu.getAccZ();
    wcount++;

    // 攒满一个 2 秒窗口就发布，然后清空重来
    if (wcount >= NSAMP) {
      publishWindow();
      wcount = 0;
    }
  }
}
