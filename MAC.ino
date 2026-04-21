#include <WiFi.h>

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA); // El modo debe estar en Estación para obtener la MAC de esa interfaz
  
  delay(1000); // Pausa para que el puerto serial se estabilice
  
  Serial.println();
  Serial.print("Dirección MAC del ESP32: ");
  Serial.println(WiFi.macAddress());
}

void loop() {
  // No necesitamos hacer nada en el bucle
}