// Gesture-controlled ESCs over Wi-Fi (Arduino UNO R4 WiFi).
//
// Receives UDP packets "<token> <nonce> <seq> <CMD>\n" from `gesturectl run` and ramps
// the throttle of all ESCs together. Safety:
//   - ESCs are armed at idle before Wi-Fi starts, and return to idle on STOP;
//   - packets with the wrong token, or stale/replayed sequence numbers, are ignored;
//   - watchdog: no valid packet for WATCHDOG_MS -> idle (sender crashed, Wi-Fi dropped...);
//   - throttle is capped at US_MAX. Raise it only deliberately, with props off first.
//
// The logic is mirrored in src/gesturectl/motor_sim.py, which the unit tests exercise.
// Keep the two in sync.
//
// Setup: copy secrets.example.h to secrets.h and fill it in (secrets.h is git-ignored).

#include <WiFiS3.h>
#include <WiFiUdp.h>
#include <Servo.h>
#include "secrets.h"  // WIFI_SSID, WIFI_PASS, CONTROL_TOKEN

const uint16_t UDP_PORT = 4210;
const uint8_t ESC_PINS[] = {9, 10, 11};
const uint8_t NUM_ESC = sizeof(ESC_PINS) / sizeof(ESC_PINS[0]);
const uint8_t LED_PIN = LED_BUILTIN;

const int US_MIN = 1000;               // armed / idle
const int US_MAX = 1200;               // bench-safety cap
const float RAMP_US_PER_S = 100.0;     // throttle change rate for UP / DOWN
const unsigned long WATCHDOG_MS = 500;

Servo escs[NUM_ESC];
WiFiUDP udp;

float throttleUs = US_MIN;
char command[8] = "STOP";
char lastNonce[17] = "";
unsigned long lastSeq = 0;
unsigned long lastPacketMs = 0;
bool everReceived = false;
unsigned long lastTickMs = 0;

void writeEscs(int us) {
  for (uint8_t i = 0; i < NUM_ESC; i++) escs[i].writeMicroseconds(us);
}

void connectWifi() {
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.begin(WIFI_SSID, WIFI_PASS) != WL_CONNECTED) {
    writeEscs(US_MIN);  // keep motors idle while (re)connecting
    Serial.print(".");
    delay(1000);
  }
  Serial.print("\nConnected, IP: ");
  Serial.println(WiFi.localIP());
  udp.begin(UDP_PORT);
}

bool isValidCommand(const char* c) {
  return !strcmp(c, "UP") || !strcmp(c, "DOWN") || !strcmp(c, "STOP") || !strcmp(c, "HOLD");
}

// Returns true if the packet was accepted.
bool handlePacket(char* buf) {
  char* token = strtok(buf, " \r\n");
  char* nonce = strtok(nullptr, " \r\n");
  char* seqStr = strtok(nullptr, " \r\n");
  char* cmd = strtok(nullptr, " \r\n");
  if (!token || !nonce || !seqStr || !cmd) return false;
  if (strcmp(token, CONTROL_TOKEN) != 0) return false;
  if (strlen(nonce) >= sizeof(lastNonce) || !isValidCommand(cmd)) return false;

  unsigned long seq = strtoul(seqStr, nullptr, 10);
  bool sameSender = strcmp(nonce, lastNonce) == 0;
  if (sameSender && seq <= lastSeq) return false;  // stale or replayed

  strcpy(lastNonce, nonce);
  lastSeq = seq;
  strncpy(command, cmd, sizeof(command) - 1);
  lastPacketMs = millis();
  everReceived = true;
  return true;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  for (uint8_t i = 0; i < NUM_ESC; i++) escs[i].attach(ESC_PINS[i]);
  writeEscs(US_MIN);
  delay(2000);  // let the ESCs arm at idle
  connectWifi();
  lastTickMs = millis();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    throttleUs = US_MIN;
    connectWifi();
  }

  int size = udp.parsePacket();
  if (size > 0) {
    char buf[96];
    int len = udp.read(buf, sizeof(buf) - 1);
    buf[len > 0 ? len : 0] = '\0';
    handlePacket(buf);
  }

  unsigned long now = millis();
  float dt = (now - lastTickMs) / 1000.0;
  lastTickMs = now;

  bool watchdogTripped = !everReceived || (now - lastPacketMs > WATCHDOG_MS);
  if (watchdogTripped) strcpy(command, "STOP");

  if (!strcmp(command, "UP")) throttleUs += RAMP_US_PER_S * dt;
  else if (!strcmp(command, "DOWN")) throttleUs -= RAMP_US_PER_S * dt;
  else if (!strcmp(command, "STOP")) throttleUs = US_MIN;
  // HOLD: keep the current throttle

  throttleUs = constrain(throttleUs, US_MIN, US_MAX);
  writeEscs((int)throttleUs);
  digitalWrite(LED_PIN, throttleUs > US_MIN ? HIGH : LOW);
  delay(5);
}
