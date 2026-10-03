/*
 * IMU 边缘端：ESP32 + MPU6050 -> WiFi -> MQTT
 *
 * 读加速度计，算幅值，做简单跌倒启发式判断，发 JSON 到 MQTT。
 *
 * 依赖库（Arduino IDE 库管理器搜索安装）：
 *   - MPU6050_tockn   (作者 tockn)
 *   - PubSubClient    (作者 Nick O'Leary)
 *   （WiFi.h 是 ESP32 自带）
 *
 * 接线：
 *   MPU6050 VCC -> 3.3V   GND -> GND   SDA -> GPIO21   SCL -> GPIO22
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <MPU6050_tockn.h>

// ============ 改成你自己的 ============
const char* WIFI_SSID   = "你的WiFi名";
const char* WIFI_PASS   = "你的WiFi密码";
const char* MQTT_BROKER = "192.168.1.100";   // 你电脑的局域网IP，cmd 里 ipconfig 查
const int   MQTT_PORT   = 1883;
const char* TOPIC       = "fall/imu/events";
const char* DEVICE      = "imu01";
// ======================================

WiFiClient   espClient;
PubSubClient client(espClient);
MPU6050      mpu(Wire);

void reconnect() {
  while (!client.connected()) {
    Serial.print("连接 MQTT...");
    if (client.connect("esp32-imu")) {
      Serial.println("成功");
    } else {
      Serial.print("失败, rc=");
      Serial.print(client.state());
      Serial.println(" 5秒后重试");
      delay(5000);
    }
  }
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
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  mpu.update();
  float ax  = mpu.getAccX();      // 单位 g
  float ay  = mpu.getAccY();
  float az  = mpu.getAccZ();
  float mag = sqrt(ax * ax + ay * ay + az * az);   // 幅值，静止约 1g

  // 简单跌倒启发：撞击尖峰(幅值 > 2.5g) 或 短暂失重(幅值 < 0.5g)
  bool fall = (mag > 2.5f || mag < 0.5f);

  char payload[160];
  snprintf(payload, sizeof(payload),
           "{\"device\":\"%s\",\"ax\":%.2f,\"ay\":%.2f,\"az\":%.2f,\"mag\":%.2f,\"fall\":%s}",
           DEVICE, ax, ay, az, mag, fall ? "true" : "false");
  client.publish(TOPIC, payload);
  Serial.println(payload);

  delay(100);   // 约 10Hz
}
