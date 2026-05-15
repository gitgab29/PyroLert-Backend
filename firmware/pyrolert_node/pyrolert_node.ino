/*
 * PyroLert Sensor Node — Production
 * ESP32 + 6x MQ + DHT22 + PMS7003 → WiFi → HTTP POST /infer
 *
 * Sends 1 JSON sample/sec to the PyroLert backend and prints
 * the fire-class prediction to Serial.
 *
 * Pinout:
 *   MQ135 → GPIO36    MQ4  → GPIO39
 *   MQ5   → GPIO34    MQ6  → GPIO35   (MQ5 pin = MQ9 sensor, firmware label kept)
 *   MQ7   → GPIO32    MQ8  → GPIO33
 *   DHT22 → GPIO19
 *   PMS7003 TX → GPIO16 (UART2 RX)
 *   PMS7003 RX → GPIO17 (UART2 TX)
 *
 * Required libraries (install via Arduino Library Manager):
 *   - ArduinoJson  >= 6.21
 *   - DHT sensor library (Adafruit)
 *   - Adafruit Unified Sensor
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// ================================================================
// User config — edit these before flashing
// ================================================================
#define WIFI_SSID       "PyroLert"
#define WIFI_PASSWORD   "pyrolert"
#define BACKEND_URL     "https://your-app-name.koyeb.app/infer"

#define SAMPLE_INTERVAL_MS  1000   // 1 Hz
#define WARMUP_SECONDS      180    // 3 min; bump to 600+ for real data collection
#define HTTP_TIMEOUT_MS     10000  // TLS handshake alone can take 2-5s on ESP32
// ================================================================

// ---------- Pins ----------
#define DHT_PIN   19
#define DHT_TYPE  DHT22

#define PIN_MQ135  36
#define PIN_MQ4    39
#define PIN_MQ5    34   // Physical sensor: MQ9 (firmware mislabel retained)
#define PIN_MQ6    35
#define PIN_MQ7    32
#define PIN_MQ8    33

// PMS7003 on UART2
#define PMS_RX  16   // ESP32 RX2  ←  PMS TX
#define PMS_TX  17   // ESP32 TX2  →  PMS RX
HardwareSerial PMSSerial(2);

// ---------- Globals ----------
DHT dht(DHT_PIN, DHT_TYPE);

struct PMSData {
  // ATM readings — what the model was trained on
  uint16_t pm1_0_atm  = 0;
  uint16_t pm2_5_atm  = 0;
  uint16_t pm10_atm   = 0;
  // Particle counts per 0.1 L of air
  uint16_t pcnt_0_3um = 0;
  uint16_t pcnt_0_5um = 0;
  uint16_t pcnt_1_0um = 0;
  uint16_t pcnt_2_5um = 0;
  uint16_t pcnt_5_0um = 0;
  uint16_t pcnt_10um  = 0;
  bool valid = false;
};

unsigned long bootMillis      = 0;
unsigned long lastSampleMs    = 0;
PMSData       latestPMS;

// Persistent TLS + HTTP objects — declared globally so the TLS session is
// reused across requests instead of re-negotiated every second.  Re-negotiating
// on every loop iteration allocates ~50 KB from the mbedTLS heap and causes
// progressive fragmentation that exhausts free memory after ~30 samples.
WiFiClientSecure tlsClient;
HTTPClient       http;

// ---------- WiFi ----------
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.disconnect(true);
  delay(200);
  Serial.printf("[WiFi] Connecting to %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - t0 > 20000) {
      Serial.println("\n[WiFi] Timeout — will retry next sample.");
      return;
    }
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected. IP: %s\n", WiFi.localIP().toString().c_str());
}

// ---------- PMS7003 parser ----------
// 32-byte frame starting with 0x42 0x4D, checksum on bytes 0-29.
// Drains the UART ring buffer on every call — must be called frequently.
bool readPMS(PMSData &out) {
  static uint8_t buf[32];
  static uint8_t idx = 0;

  while (PMSSerial.available()) {
    uint8_t b = PMSSerial.read();
    if (idx == 0 && b != 0x42) continue;
    if (idx == 1 && b != 0x4D) { idx = 0; continue; }
    buf[idx++] = b;

    if (idx == 32) {
      idx = 0;
      uint16_t chk = 0;
      for (int i = 0; i < 30; i++) chk += buf[i];
      if (chk != ((buf[30] << 8) | buf[31])) return false;

      // Atmospheric readings (bytes 10-15)
      out.pm1_0_atm  = (buf[10] << 8) | buf[11];
      out.pm2_5_atm  = (buf[12] << 8) | buf[13];
      out.pm10_atm   = (buf[14] << 8) | buf[15];
      // Particle counts (bytes 16-27)
      out.pcnt_0_3um = (buf[16] << 8) | buf[17];
      out.pcnt_0_5um = (buf[18] << 8) | buf[19];
      out.pcnt_1_0um = (buf[20] << 8) | buf[21];
      out.pcnt_2_5um = (buf[22] << 8) | buf[23];
      out.pcnt_5_0um = (buf[24] << 8) | buf[25];
      out.pcnt_10um  = (buf[26] << 8) | buf[27];
      out.valid = true;
      return true;
    }
  }
  return false;
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n[PyroLert] Booting...");

  analogReadResolution(12);         // 12-bit ADC, 0-4095
  analogSetAttenuation(ADC_11db);   // 0-3.3 V range

  dht.begin();
  PMSSerial.begin(9600, SERIAL_8N1, PMS_RX, PMS_TX);

  connectWiFi();

  // Configure the persistent TLS client once — reused for every POST.
  tlsClient.setInsecure();
  http.setReuse(true);

  bootMillis = millis();
  Serial.printf("[PyroLert] Warming up for %d s. Predictions will return NONE until ready.\n",
                WARMUP_SECONDS);
}

// ---------- Loop ----------
void loop() {
  // Drain PMS UART continuously so we always have a fresh frame
  readPMS(latestPMS);

  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) return;

  // ---- Reconnect if needed ----
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Lost — reconnecting...");
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) return; // skip sample, try next second
  }

  // ---- Read sensors ----
  unsigned long uptimeSec = (now - bootMillis) / 1000;
  bool warmedUp = (uptimeSec >= WARMUP_SECONDS);

  int mq135 = analogRead(PIN_MQ135);
  int mq4   = analogRead(PIN_MQ4);
  int mq5   = analogRead(PIN_MQ5);
  int mq6   = analogRead(PIN_MQ6);
  int mq7   = analogRead(PIN_MQ7);
  int mq8   = analogRead(PIN_MQ8);

  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();
  if (isnan(temp)) temp = -999.0f;
  if (isnan(hum))  hum  = -999.0f;

  // ---- Build JSON body ----
  // Field names must match the backend exactly (see inference/features.py)
  StaticJsonDocument<512> doc;
  doc["mq135"]        = mq135;
  doc["mq4"]          = mq4;
  doc["mq5"]          = mq5;   // MQ9 sensor, mq5 label kept for model compatibility
  doc["mq6"]          = mq6;
  doc["mq7"]          = mq7;
  doc["mq8"]          = mq8;
  doc["temperature_c"]= serialized(String(temp, 2));
  doc["humidity_pct"] = serialized(String(hum,  2));
  doc["pm1_0_atm"]    = latestPMS.pm1_0_atm;
  doc["pm2_5_atm"]    = latestPMS.pm2_5_atm;
  doc["pm10_atm"]     = latestPMS.pm10_atm;
  doc["pcnt_0_3um"]   = latestPMS.pcnt_0_3um;
  doc["pcnt_0_5um"]   = latestPMS.pcnt_0_5um;
  doc["pcnt_1_0um"]   = latestPMS.pcnt_1_0um;
  doc["pcnt_2_5um"]   = latestPMS.pcnt_2_5um;
  doc["pcnt_5_0um"]   = latestPMS.pcnt_5_0um;
  doc["pcnt_10um"]    = latestPMS.pcnt_10um;
  doc["pms_valid"]    = latestPMS.valid ? 1 : 0;
  doc["warmed_up"]    = warmedUp ? 1 : 0;

  String body;
  serializeJson(doc, body);

  // ---- POST to backend ----
  http.begin(tlsClient, BACKEND_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(HTTP_TIMEOUT_MS);

  int httpCode = http.POST(body);

  if (httpCode == 200) {
    String raw = http.getString();

    StaticJsonDocument<256> resp;
    DeserializationError err = deserializeJson(resp, raw);
    if (!err) {
      const char* pred   = resp["prediction"] | "?";
      const char* reason = resp["reason"]     | "";

      if (strlen(reason) > 0) {
        // warmup / pms_invalid / low_confidence
        Serial.printf("[Infer] %-4s  reason=%-15s  uptime=%lus\n",
                      pred, reason, uptimeSec);
      } else {
        float s1   = resp["s1_proba"]  | -1.0f;
        float conf = resp["confidence"]| -1.0f;
        Serial.printf("[Infer] %-4s  s1=%.3f  conf=%.3f  uptime=%lus\n",
                      pred, s1, conf, uptimeSec);
      }
    } else {
      Serial.printf("[HTTP] Bad JSON: %s\n", raw.c_str());
    }
  } else {
    Serial.printf("[HTTP] POST failed, code=%d\n", httpCode);
    // Force a fresh TLS handshake next iteration — the connection may be
    // stale (WiFi reconnect, Render keep-alive timeout, cold start).
    http.end();
    tlsClient.stop();
    lastSampleMs = millis();
    return;
  }

  http.end();
  // Timestamp measured after the blocking POST so the next sample fires
  // SAMPLE_INTERVAL_MS after this one completes, not after it started.
  lastSampleMs = millis();

  // ---- Debug echo ----
  Serial.printf("[Raw]  mq135=%4d mq4=%4d mq5=%4d mq6=%4d mq7=%4d mq8=%4d"
                "  t=%.1f h=%.1f  pm25=%u  pms=%d warmed=%d\n",
                mq135, mq4, mq5, mq6, mq7, mq8,
                temp, hum, latestPMS.pm2_5_atm,
                latestPMS.valid ? 1 : 0, warmedUp ? 1 : 0);
}
