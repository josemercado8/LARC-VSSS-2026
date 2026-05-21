# =============================================================================
# CALIBRADOR ROBÓTICO - HERRAMIENTA DE CONFIGURACIÓN INICIAL (VERSIÓN HSV)
# =============================================================================
# Desarrollado por: M. en C. José Manuel Mercado Blanco

# Este programa debe ejecutarse antes del sistema de control principal.
# Permite:
#   1. Seleccionar el equipo (azul o amarillo) -> guarda IDs de ArUco.
#   2. Seleccionar las 4 esquinas del campo -> guarda coordenadas para enderezar.
#   3. Calibrar el color naranja de la pelota (clic en la pelota) -> guarda rangos HSV.
#   4. Calibrar la cámara ojo de pez (dos métodos) -> guarda matriz de cámara y coeficientes.
#   5. Ajustar parámetros de cámara (exposición, enfoque) en tiempo real.
#   6. Usar configuración de cámara por defecto (sin corrección de distorsión).
# =============================================================================

import cv2
import numpy as np
import json
import time
import os
from scipy.optimize import minimize

class CalibradorRobotico:
    """
    Clase principal que contiene todos los métodos y estados del calibrador.
    """
    def __init__(self):
        # =========================================================
        # Zona de Configuración de ROL
        # =========================================================
        self.team_data = {
            'amarillo': {'POR': 771, 'MED': 955, 'DEL': 939},
            'azul':     {'POR': 273, 'MED': 256, 'DEL': 272}
        }
        # =========================================================

        self.selected_team = None
        self.calibration_colors = ['naranja']
        self.hsv_ranges = {}
        self.points = []
        self.state = 'select_team'
        self.raw_frame = None

        # Márgenes independientes para el espacio HSV (OpenCV: H 0-179, S 0-255, V 0-255)
        self.H_MARGIN = 12  # Margen estrecho para el tono/color (naranja)      10
        self.S_MARGIN = 70  # Margen amplio para la saturación (tolerancia a pérdida de color)  90
        self.V_MARGIN = 70  # Margen amplio para el valor (tolerancia a sombras/luces)  100

        # --- Detector de ArUco ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # --- Ecualizador de histograma para mejorar detección bajo iluminación variable ---
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # --- Calibración de cámara (ojo de pez) ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None
        self.calibration_file = "camera_calibration.json"
        self.load_camera_calibration()   # Intenta cargar archivo existente

        # --- Parámetros de cámara por defecto (se actualizan con sliders) ---
        self.camera_params = {
            'exposure': 0,
            'focus': 0,
            'auto_exposure': -6,
            'auto_wb': 0,
            'auto_focus': 0
        }

    # ------------------------------------------------------------------
    # Aplicar configuración fija de cámara (exposición -4)
    # ------------------------------------------------------------------
    def apply_fixed_camera_settings(self, cap):
        """Aplica exposición manual -4 y desactiva modos automáticos."""
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)   # 0 = manual
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        print("Configuración de cámara aplicada: Exposición = -4, Auto off.")

    # ------------------------------------------------------------------
    # Carga de calibración de cámara existente
    # ------------------------------------------------------------------
    def load_camera_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
                self.dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float32)
                h, w = data['image_size']
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    self.camera_matrix, self.dist_coeffs, np.eye(3), self.camera_matrix, (w, h), cv2.CV_32FC1
                )
                print("Calibración de cámara cargada.")
                return True
            except:
                print("Error al cargar calibración.")
        return False

    # ------------------------------------------------------------------
    # Restablecer configuración por defecto de cámara
    # ------------------------------------------------------------------
    def reset_camera_to_default(self):
        """Elimina cualquier calibración de distorsión, usando la cámara tal cual."""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None
        print("Configuración de cámara por defecto (sin corrección de distorsión).")

    def undistort_frame(self, frame):
        if self.map1 is not None and self.map2 is not None:
            return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        return frame

    # ------------------------------------------------------------------
    # OPCIÓN 3: Calibración usando puntos de la cancha (líneas rectas)
    # ------------------------------------------------------------------
    def calibrate_with_field_points(self, cap):
        print("\n=== CALIBRACIÓN USANDO PUNTOS DE LA CANCHA ===")
        print("Vamos a seleccionar puntos sobre líneas rectas del campo.")
        print("Instrucciones:")
        print("1. Haz clic en puntos que pertenezcan a una misma línea recta (por ejemplo, borde de la banda).")
        print("2. Presiona 'n' para empezar una nueva línea.")
        print("3. Presiona 'c' para calcular la calibración con las líneas ingresadas.")
        print("4. Necesitas al menos 2 líneas con 3 o más puntos cada una.")

        lines_points = []
        current_line = []

        def mouse_callback_lines(event, x, y, flags, param):
            nonlocal current_line
            if event == cv2.EVENT_LBUTTONDOWN:
                current_line.append((x, y))
                print(f"Punto añadido a línea actual: ({x},{y})")

        cv2.namedWindow("Selecciona puntos de la cancha")
        cv2.setMouseCallback("Selecciona puntos de la cancha", mouse_callback_lines)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()

            for line in lines_points:
                for pt in line:
                    cv2.circle(display, pt, 3, (0,255,0), -1)
                if len(line) >= 2:
                    for i in range(len(line)-1):
                        cv2.line(display, line[i], line[i+1], (0,255,0), 2)

            for pt in current_line:
                cv2.circle(display, pt, 3, (0,0,255), -1)
            if len(current_line) >= 2:
                for i in range(len(current_line)-1):
                    cv2.line(display, current_line[i], current_line[i+1], (0,0,255), 2)

            cv2.putText(display, f"Lineas: {len(lines_points)}  Puntos linea actual: {len(current_line)}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(display, "n: nueva linea   c: calcular   q: cancelar",
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Selecciona puntos de la cancha", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                if len(current_line) >= 2:
                    lines_points.append(current_line.copy())
                    print(f"Línea guardada con {len(current_line)} puntos.")
                else:
                    print("Cada línea necesita al menos 2 puntos.")
                current_line = []
            elif key == ord('c'):
                if len(current_line) >= 2:
                    lines_points.append(current_line.copy())
                    print(f"Última línea guardada con {len(current_line)} puntos.")
                if len(lines_points) < 2:
                    print("Necesitas al menos 2 líneas con puntos.")
                    continue
                success = self.optimize_distortion_from_lines(frame.shape[:2], lines_points)
                if success:
                    cv2.destroyWindow("Selecciona puntos de la cancha")
                    return True
                else:
                    print("Falló la optimización. Intenta con más puntos o líneas.")
            elif key == ord('q'):
                cv2.destroyWindow("Selecciona puntos de la cancha")
                return False

    def optimize_distortion_from_lines(self, img_shape, lines_points):
        h, w = img_shape[:2]
        K0 = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        D0 = np.zeros(4)

        def cost(params):
            K = K0.copy()
            D = params[:4].astype(np.float64)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
            total_error = 0.0
            for line in lines_points:
                pts_undist = []
                for (x, y) in line:
                    u = map1[y, x]
                    v = map2[y, x]
                    pts_undist.append([u, v])
                pts_undist = np.array(pts_undist)
                if len(pts_undist) < 2:
                    continue
                x_coords = pts_undist[:, 0]
                y_coords = pts_undist[:, 1]
                if np.std(x_coords) < 1e-6:
                    mean_x = np.mean(x_coords)
                    error = np.sum((x_coords - mean_x)**2)
                else:
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                    y_pred = m * x_coords + b
                    error = np.sum((y_coords - y_pred)**2)
                total_error += error
            return total_error

        print("Optimizando parámetros de distorsión... (puede tardar unos segundos)")
        res = minimize(cost, D0, method='Powell', options={'xtol': 1e-4, 'ftol': 1e-4})
        if res.success:
            D_opt = res.x
            print(f"Optimización exitosa. Coeficientes: k1={D_opt[0]:.4f}, k2={D_opt[1]:.4f}, k3={D_opt[2]:.4f}, k4={D_opt[3]:.4f}")
            self.camera_matrix = K0.astype(np.float32)
            self.dist_coeffs = D_opt.astype(np.float32).reshape(4, 1)
            h, w = img_shape
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, np.eye(3), self.camera_matrix, (w, h), cv2.CV_32FC1
            )
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'image_size': [w, h]
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Calibración guardada en {self.calibration_file}")
            return True
        else:
            print("Falló la optimización:", res.message)
            return False

    # ------------------------------------------------------------------
    # OPCIÓN 2: Ajuste manual de distorsión con sliders
    # ------------------------------------------------------------------
    def manual_fisheye_calibration(self, cap):
        print("\n=== CALIBRACIÓN MANUAL CON SLIDERS ===")
        print("Mueve los sliders (k1..k4) hasta que las líneas de la cancha se vean rectas.")
        print("Presiona 's' para guardar la calibración, 'q' para salir sin guardar.")

        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer la cámara.")
            return False
        h, w = frame.shape[:2]
        K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        D = np.zeros(4)

        cv2.namedWindow('Ajuste manual de distorsión')
        cv2.createTrackbar('k1', 'Ajuste manual de distorsión', 500, 1000, lambda x: None)
        cv2.createTrackbar('k2', 'Ajuste manual de distorsión', 500, 1000, lambda x: None)
        cv2.createTrackbar('k3', 'Ajuste manual de distorsión', 500, 1000, lambda x: None)
        cv2.createTrackbar('k4', 'Ajuste manual de distorsión', 500, 1000, lambda x: None)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            k1 = (cv2.getTrackbarPos('k1', 'Ajuste manual de distorsión') - 500) / 100.0
            k2 = (cv2.getTrackbarPos('k2', 'Ajuste manual de distorsión') - 500) / 100.0
            k3 = (cv2.getTrackbarPos('k3', 'Ajuste manual de distorsión') - 500) / 100.0
            k4 = (cv2.getTrackbarPos('k4', 'Ajuste manual de distorsión') - 500) / 100.0
            D[:] = [k1, k2, k3, k4]

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
            undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            cv2.imshow('Ajuste manual de distorsión', undist)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.camera_matrix = K
                self.dist_coeffs = D.reshape(4, 1)
                self.map1, self.map2 = map1, map2
                cal_data = {
                    'camera_matrix': K.tolist(),
                    'dist_coeffs': D.tolist(),
                    'image_size': [w, h]
                }
                with open(self.calibration_file, 'w') as f:
                    json.dump(cal_data, f)
                print("Calibración manual guardada.")
                cv2.destroyWindow('Ajuste manual de distorsión')
                return True
            elif key == ord('q'):
                cv2.destroyWindow('Ajuste manual de distorsión')
                return False

    # ------------------------------------------------------------------
    # Método principal de calibración de cámara (elige método)
    # ------------------------------------------------------------------
    def calibrate_camera(self, cap):
        print("\n--- CALIBRACIÓN DE CÁMARA (ojo de pez) ---")
        print("Selecciona el método:")
        print("  [p] Calibración usando puntos de la cancha (recomendado)")
        print("  [m] Ajuste manual con sliders")
        print("  [q] Cancelar")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('p'):
                return self.calibrate_with_field_points(cap)
            elif key == ord('m'):
                return self.manual_fisheye_calibration(cap)
            elif key == ord('q'):
                return False

    # ------------------------------------------------------------------
    # Ajuste de parámetros de cámara (exposición, enfoque, etc.)
    # ------------------------------------------------------------------
    def adjust_camera_parameters(self, cap):
        print("\n=== AJUSTE DE PARÁMETROS DE CÁMARA ===")
        print("Desactivando modos automáticos...")

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, self.camera_params['exposure']) 

        current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        current_focus = cap.get(cv2.CAP_PROP_FOCUS)
        print(f"Exposición actual: {current_exposure}")
        print(f"Enfoque actual: {current_focus}")

        cv2.namedWindow('Parametros de Camara')
        initial_exp_track = int((current_exposure + 13) * 100 / 12) if current_exposure else 50
        initial_exp_track = max(0, min(100, initial_exp_track))
        cv2.createTrackbar('Exposicion', 'Parametros de Camara', initial_exp_track, 100, lambda x: None)
        initial_focus = int(current_focus) if current_focus > 0 else 0
        cv2.createTrackbar('Enfoque', 'Parametros de Camara', initial_focus, 255, lambda x: None)

        print("Usa los sliders para ajustar Exposición y Enfoque.")
        print("Presiona 's' para guardar los valores y salir.")
        print("Presiona 'q' para cancelar sin guardar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            exp_track = cv2.getTrackbarPos('Exposicion', 'Parametros de Camara')
            focus_val = cv2.getTrackbarPos('Enfoque', 'Parametros de Camara')
            exposure_val = -13 + int((exp_track / 100.0) * 12)

            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)

            info_frame = frame.copy()
            cv2.putText(info_frame, f"Exposicion: {exposure_val}  (track: {exp_track})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(info_frame, f"Enfoque: {focus_val}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(info_frame, "s: guardar   q: cancelar",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow('Parametros de Camara', info_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.camera_params['exposure'] = exposure_val
                self.camera_params['focus'] = focus_val
                self.camera_params['auto_exposure'] = 0
                self.camera_params['auto_wb'] = 0
                self.camera_params['auto_focus'] = 0
                print(f"Parámetros guardados: exposición={exposure_val}, enfoque={focus_val}")
                cv2.destroyWindow('Parametros de Camara')
                return True
            elif key == ord('q'):
                cv2.destroyWindow('Parametros de Camara')
                return False

    # ------------------------------------------------------------------
    # Guardar configuración de equipo
    # ------------------------------------------------------------------
    def save_team_config(self, team_color):
        self.selected_team = team_color
        enemy_team = 'amarillo' if self.selected_team == 'azul' else 'azul'
        config_data = {
            'equipo_seleccionado': self.selected_team,
            'aliados': self.team_data[self.selected_team],
            'equipo_contrario': enemy_team,
            'contrarios': self.team_data[enemy_team]
        }
        try:
            with open('equipo_config.json', 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuración guardada - Aliados: {config_data['aliados']}")
            self.state = 'select_field'
        except Exception as e:
            print(f"Error al guardar el equipo: {e}")

    # ------------------------------------------------------------------
    # Callback del mouse (maneja clics según el estado actual)
    # ------------------------------------------------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == 'select_field':
                if len(self.points) < 4:
                    self.points.append((x, y))
                if len(self.points) == 4:
                    self.state = 'calibrate'
                    print("Campo seleccionado. Guardando configuración...")
                    try:
                        with open('cancha_config.json', 'w') as f:
                            json.dump({'puntos_cancha': self.points}, f, indent=2)
                        print("Cancha guardada exitosamente.")
                    except Exception as e:
                        print(f"Error al guardar: {e}")

            elif self.state == 'calibrate':
                if self.raw_frame is not None:
                    # CONVERSIÓN A HSV: Convertimos todo el frame BGR a HSV
                    hsv_frame = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2HSV)
                    hsv_pixel = hsv_frame[y, x]   # [H, S, V]
                    
                    # Extraer canales
                    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

                    # Calcular rangos aplicando márgenes específicos y asegurando límites 
                    # de OpenCV (H: 0-179, S: 0-255, V: 0-255)
                    lower = [max(h - self.H_MARGIN, 0),
                             max(s - self.S_MARGIN, 0),
                             max(v - self.V_MARGIN, 0)]
                             
                    upper = [min(h + self.H_MARGIN, 179),
                             min(s + self.S_MARGIN, 255),
                             min(v + self.V_MARGIN, 255)]

                    color_name = self.calibration_colors[len(self.hsv_ranges)]
                    self.hsv_ranges[color_name] = {'lower': lower, 'upper': upper}
                    print(f"Calibrado {color_name}: HSV {[h,s,v]} -> lower: {lower}, upper: {upper}")

                    if len(self.hsv_ranges) == len(self.calibration_colors):
                        # Guardamos en un archivo de configuración HSV
                        with open('hsv_calibration.json', 'w') as f:
                            json.dump(self.hsv_ranges, f, indent=2)
                        print("¡Calibración del color naranja guardada en hsv_calibration.json!")
                        self.state = 'done'

    # ------------------------------------------------------------------
    # Bucle principal (Máquina de estados)
    # ------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(1)   # Cambiar índice de cámara si es necesario

        self.apply_fixed_camera_settings(cap)

        cv2.namedWindow('Calibrador y Detector')
        cv2.setMouseCallback('Calibrador y Detector', self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            frame_undistorted = self.undistort_frame(frame)
            self.raw_frame = frame_undistorted.copy()
            display = frame_undistorted.copy()

            # --- Detección de ArUco con ecualización de histograma ---
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            gray_eq = self.clahe.apply(gray)          # CLAHE para mejorar contraste local
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray_eq)
            # ---------------------------------------------------------

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                cv2.putText(display, f"Robots: {len(ids)}", (display.shape[1] - 150, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- MÁQUINA DE ESTADOS ---
            if self.state == 'select_team':
                overlay = display.copy()
                cv2.rectangle(overlay, (10, 10), (520, 190), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                cv2.putText(display, "SELECCIONA TU EQUIPO:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, "Presiona '1' para AZUL", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)
                cv2.putText(display, "Presiona '2' para AMARILLO", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, "Presiona 'c' para CALIBRAR CAMARA", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(display, "Presiona 'p' para AJUSTAR PARAMETROS", (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display, "Presiona 'd' para usar CAMARA POR DEFECTO", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            elif self.state == 'select_field':
                text = f"EQUIPO {self.selected_team.upper()}: Clic en 4 esquinas"
                cv2.putText(display, text, (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                for p in self.points:
                    cv2.circle(display, p, 5, (0,255,0), -1)

            elif self.state == 'calibrate':
                if len(self.points) == 4:
                    pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.rectangle(display, (10, 10), (280, 100), (0, 0, 0), -1)
                
                # MODIFICADO TEXTO DE UI
                cv2.putText(display, "CALIBRACION PELOTA (HSV)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if len(self.hsv_ranges) == 0:
                    cv2.putText(display, "-> HAZ CLIC EN LA PELOTA <-", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            elif self.state == 'done':
                if len(self.points) == 4:
                    pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(display, "TODO GUARDADO. Presiona 'q' para salir.", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow('Calibrador y Detector', display)

            # --- Manejo de teclas ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if self.state == 'select_team':
                if key == ord('1'):
                    self.save_team_config('azul')
                elif key == ord('2'):
                    self.save_team_config('amarillo')
                elif key == ord('c'):
                    if self.calibrate_camera(cap):
                        print("Calibración completada. Ahora selecciona equipo.")
                    else:
                        print("Calibración cancelada.")
                elif key == ord('p'):
                    self.adjust_camera_parameters(cap)
                elif key == ord('d'):
                    self.reset_camera_to_default()
                    print("Usando cámara sin corrección de distorsión.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    calibrador = CalibradorRobotico()
    calibrador.run()
