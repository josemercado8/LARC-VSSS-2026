#include <WiFi.h>
#include <esp_now.h>

// ==========================================
// DIRECCIONES MAC DE LOS 3 ROBOTS (ESCLAVOS)
// ==========================================
uint8_t mac_DEL[] = {0x94, 0x54, 0xC5, 0x63, 0xBA, 0xF8};
uint8_t mac_POR[] = {0x5C, 0x01, 0x3B, 0x68, 0x6D, 0x94};
uint8_t mac_MED[] = {0x5C, 0x01, 0x3B, 0x67, 0x65, 0x2C};

esp_now_peer_info_t peerInfo;
String inputString = "";
bool stringComplete = false;

// Contadores para depuración (opcional)
unsigned long lastPrintTime = 0;
int packetsSent = 0;
int packetsFailed = 0;

// Callback para confirmación de envío (firma para core v3.x)
void OnDataSent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    packetsSent++;
  } else {
    packetsFailed++;
  }
}

// Función para registrar esclavo
void registrarEsclavo(uint8_t *mac) {
  memcpy(peerInfo.peer_addr, mac, 6);
  peerInfo.channel = 0;          // Usará el canal configurado en WiFi
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Error añadiendo peer");
  } else {
    Serial.print("Peer registrado: ");
    for (int i = 0; i < 6; i++) {
      Serial.printf("%02X", mac[i]);
      if (i < 5) Serial.print(":");
    }
    Serial.println();
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(11);   // Fijar canal 1 (debe ser el mismo en los esclavos)

  if (esp_now_init() != ESP_OK) {
    Serial.println("ERR:INIT_ESPNOW");
    return;
  }

  // Registrar callback de envío (opcional, pero si lo usas debe tener la firma correcta)
  // esp_now_register_send_cb(OnDataSent);  // Descomenta si quieres estadísticas

  registrarEsclavo(mac_POR);
  registrarEsclavo(mac_MED);
  registrarEsclavo(mac_DEL);

  Serial.println("Maestro listo. Esperando PWMs desde Python...");
  Serial.println("Formato esperado: <pwmL,pwmR|pwmL,pwmR|pwmL,pwmR>");
}

void loop() {
  // --- LEER DATOS DEL PUERTO SERIE (desde Python) ---
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
      break;
    } else {
      inputString += inChar;
    }
  }

  // --- PROCESAR TRAMA ---
  if (stringComplete) {
    inputString.trim();
    // Depuración: mostrar trama recibida
    Serial.print("Trama recibida: ");
    Serial.println(inputString);

    if (inputString.startsWith("<") && inputString.endsWith(">")) {
      String datos = inputString.substring(1, inputString.length() - 1);
      int split1 = datos.indexOf('|');
      int split2 = datos.indexOf('|', split1 + 1);

      if (split1 > 0 && split2 > split1) {
        String datos_POR = datos.substring(0, split1);
        String datos_MED = datos.substring(split1 + 1, split2);
        String datos_DEL = datos.substring(split2 + 1);

        uint8_t* macs[3] = {mac_POR, mac_MED, mac_DEL};
        String datos[3] = {datos_POR, datos_MED, datos_DEL};
        const char* nombres[3] = {"POR", "MED", "DEL"};

        for (int i = 0; i < 3; i++) {
          datos[i].trim();
          if (datos[i].length() > 0) {
            esp_err_t result = esp_now_send(macs[i], (uint8_t*)datos[i].c_str(), datos[i].length() + 1);
            if (result == ESP_OK) {
              Serial.printf("Enviado a %s: %s\n", nombres[i], datos[i].c_str());
            } else {
              Serial.printf("Fallo en envío a %s (código: %d)\n", nombres[i], result);
            }
          } else {
            Serial.printf("Datos vacíos para %s\n", nombres[i]);
          }
        }

        // Mostrar estadísticas cada 5 segundos (solo si tienes el callback activo)
        if (millis() - lastPrintTime > 5000) {
          Serial.printf("Paquetes OK: %d | Fallos: %d\n", packetsSent, packetsFailed);
          lastPrintTime = millis();
        }
      } else {
        Serial.println("Formato inválido (faltan '|')");
      }
    } else {
      Serial.println("Trama no tiene formato <...>");
    }

    inputString = "";
    stringComplete = false;
  }
}