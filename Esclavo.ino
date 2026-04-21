#include <esp_now.h>
#include <WiFi.h>

// ===== CONFIGURACIÓN DE PINES =====
const int ENA = 4;   // PWM canal izquierdo
const int IN1 = 17;
const int IN2 = 16;
const int ENB = 23;  // PWM canal derecho
const int IN3 = 19;
const int IN4 = 22;
const int STBY = 18;

// ===== CONFIGURACIÓN DEL PWM (v3.x) =====
const int pwmFreq = 5000;     // 5 kHz
const int pwmResolution = 8;  // 0-255 (8 bits)

// ===== SEGURIDAD (WATCHDOG) =====
unsigned long lastRecvTime = 0;
const unsigned long TIMEOUT_MS = 250;   // medio segundo sin datos → parar
bool isMoving = false;

// ===== FUNCIÓN DE CONTROL DE MOTORES =====
void controlMotors(int pwmL, int pwmR) {
  digitalWrite(STBY, HIGH);

  // Motor izquierdo
  if (pwmL >= 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  }
  // Escribimos el PWM directamente en el pin (v3.x)
  ledcWrite(ENA, constrain(abs(pwmL), 0, 255));

  // Motor derecho
  if (pwmR >= 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  // Escribimos el PWM directamente en el pin (v3.x)
  ledcWrite(ENB, constrain(abs(pwmR), 0, 255));

  isMoving = (pwmL != 0 || pwmR != 0);
}

// ===== CALLBACK QUE SE EJECUTA AL RECIBIR UN PAQUETE =====
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  // Actualizar watchdog
  lastRecvTime = millis();

  // Copiar datos a un string terminado en nulo
  char data[len + 1];
  memcpy(data, incomingData, len);
  data[len] = '\0';

  String msg = String(data);
  int coma = msg.indexOf(',');

  // Depuración: mostrar lo recibido
  Serial.printf("Recibido (%d bytes): %s\n", len, data);

  if (coma > 0 && msg.length() > coma + 1) {
    int vel_L = msg.substring(0, coma).toInt();
    int vel_R = msg.substring(coma + 1).toInt();
    controlMotors(vel_L, vel_R);
    Serial.printf("Motores -> L:%d  R:%d\n", vel_L, vel_R);
  } else {
    Serial.println("Formato incorrecto, deteniendo motores");
    controlMotors(0, 0);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Iniciando esclavo ESP-NOW...");

  // Configurar pines
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(STBY, OUTPUT);

  // ===== CONFIGURAR PWM (v3.x) =====
  // ledcAttach(pin, frecuencia, resolución)
  if (!ledcAttach(ENA, pwmFreq, pwmResolution)) {
    Serial.println("Error al configurar PWM en ENA");
  }
  if (!ledcAttach(ENB, pwmFreq, pwmResolution)) {
    Serial.println("Error al configurar PWM en ENB");
  }

  // Inicializar WiFi en modo estación y fijar el mismo canal que el maestro
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(11);   // Importante: debe coincidir con el maestro

  if (esp_now_init() == ESP_OK) {
    esp_now_register_recv_cb(OnDataRecv);
    Serial.println("ESP-NOW inicializado correctamente, esperando datos...");
  } else {
    Serial.println("ERROR: No se pudo inicializar ESP-NOW");
  }

  // Detener motores al inicio
  controlMotors(0, 0);
}

void loop() {
  // Watchdog: si ha pasado el timeout y el robot se estaba moviendo, frenar por seguridad
  if (isMoving && (millis() - lastRecvTime > TIMEOUT_MS)) {
    Serial.println("⚠️ TIMEOUT: pérdida de conexión con el maestro. Frenando.");
    controlMotors(0, 0);
  }
  delay(10);   // pequeño retardo para no saturar el CPU
}