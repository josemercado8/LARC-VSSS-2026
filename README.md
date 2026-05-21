# 🤖⚽ Sistema de Control Centralizado para Robot Fútbol

Este repositorio contiene la arquitectura de software completa para un sistema de fútbol de robots autónomos. El sistema opera de manera centralizada: una computadora principal procesa la visión de la cancha mediante una cámara cenital, calcula la física y las tácticas de juego, y transmite comandos de movimiento a los robots físicos en tiempo real.

## 🏗️ Arquitectura del Sistema

El proyecto está dividido en cuatro bloques principales que trabajan en conjunto para lograr la autonomía de los robots:

### 1. 💻 Control Central y Visión por Computadora (Script Principal)
Es el "cerebro" del sistema, ejecutado en una computadora. Sus funciones son:
* **Visión Artificial:** Analiza la imagen de la cámara en tiempo real. Detecta la posición de la pelota usando filtros de color (HSV) e identifica la posición, orientación y equipo de cada robot mediante marcadores fiduciales (ArUco).
* **Inteligencia Táctica:** Asigna roles dinámicos (Portero, Medio, Delantero) basándose en la distancia a la pelota. Calcula trayectorias de intercepción, evasión de obstáculos (geocerca y repulsión de bordes) y recuperación si un robot se pierde de vista.
* **Control de Movimiento:** Utiliza controladores PD/PID para calcular la velocidad lineal y angular exacta que cada robot necesita para llegar a su objetivo.
* **Generación de Trama:** Empaqueta las velocidades de los motores izquierdo y derecho de todo el equipo en una cadena de texto (ej. `<PWM_L,PWM_R|PWM_L,PWM_R|PWM_L,PWM_R>\n`) y la envía por puerto serie.

### 2. 📡 Módulo Maestro (Transmisor)
Este código se ejecuta en un microcontrolador (como un ESP32 o Arduino) conectado directamente por USB a la computadora principal.
* Actúa como un puente de comunicación de ultra baja latencia.
* Su única tarea es escuchar el puerto serie, recibir la cadena de texto con las velocidades generadas por el "Control Central", y transmitirla de forma inalámbrica (vía radiofrecuencia, Bluetooth o Wi-Fi/ESP-NOW) hacia el campo de juego.

### 3. 🤖 Módulo Esclavo (Receptores en los Robots)
Este código vive en los microcontroladores montados sobre cada uno de los robots de la cancha.
* Están constantemente escuchando la señal inalámbrica enviada por el Módulo Maestro.
* Al recibir la trama de datos, el código esclavo la "desempaqueta" y extrae únicamente el par de velocidades (Motor Izquierdo y Motor Derecho) que le corresponde a su identificador o rol.
* Traduce esos valores numéricos en señales PWM (Modulación por Ancho de Pulso) para enviarlas al puente H (driver de motores), ejecutando físicamente el avance, retroceso o giro dictado por la computadora.

### 4. ⚙️ Herramientas de Calibración
Un conjunto de scripts auxiliares que se utilizan antes de iniciar un partido para asegurar que el sistema "vea" correctamente el entorno físico.
* **Calibrador HSV:** Permite ajustar los umbrales de luz y color para que el sistema reconozca la pelota bajo las condiciones de iluminación actuales de la sala. Genera el archivo de configuración de colores.
* **Calibrador de Cámara:** Calcula y corrige la distorsión del lente de la cámara (efecto ojo de pez) mediante un tablero de ajedrez, para que las medidas de distancia en pantalla coincidan con la realidad geométrica.
* **Configurador de Geometría:** Permite mapear los vértices de la cancha real al inicio del script principal para generar una transformación de perspectiva (cancha perfectamente rectangular en el software).

## 🔄 Flujo de Trabajo en Tiempo Real

1. **Ojos:** La cámara cenital captura un *frame* de la cancha física.
2. **Cerebro:** El **Script Principal** procesa el *frame*, ubica objetos, decide la jugada y calcula la potencia de cada llanta.
3. **Puente:** El **Maestro** recibe las instrucciones por cable y las grita por el aire.
4. **Músculo:** Los **Esclavos** escuchan el mensaje, encienden sus motores con la potencia exacta solicitada y mueven el robot.
5. *El ciclo se repite decenas de veces por segundo.*
