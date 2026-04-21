"""
SISTEMA DE CONTROL PARA ROBOT FUTBOL - VISIÓN POR COMPUTADORA
Versión con control simultáneo PD/PID (avance y giro al mismo tiempo).
Ganancias adaptativas suave-lejos / agresivo-cerca.
Prioridad posición real pelota, predictor robots, exposición fija.
MODIFICADO PARA USAR CONFIGURACIÓN HSV (calibrador)

NUEVAS FUNCIONALIDADES:
- Repulsión activa de bordes de la cancha.
- Alternar equipo (amarillo/azul) con tecla 'e' (compatible con JSON de estructura plana).
- Recuperación automática de robots perdidos (giro lento).
"""

import cv2
import numpy as np
import json
import sys
import os
import serial
import time
import argparse

# =============================================================================
# CONFIGURACIÓN DE COMPORTAMIENTO Y CONTROL
# =============================================================================
CLEARANCE_DIST = 40
RADIO_COLISION = 35
MARGEN_AREA = 10
DISTANCIA_DETRAS_PELOTA = 20
DIST_ACTIVACION_ATAQUE = 60
FUERZA_REPULSION_MAX = 80

# --- Repulsión de bordes de la cancha ---
BORDE_DIST_SEGURIDAD = 30
FUERZA_REPULSION_BORDE = 90

PRIORIDAD_DELANTERO_THRESHOLD = 1.3
APOYO_DEFENSIVE_X_FACTOR = 0.35
APOYO_INTERCEPT_DIST = 30

# Ganancias base según distancia (lejos / cerca)
KP_DIST_BASE_LEJOS = 1.8
KD_DIST_BASE_LEJOS = 0.3
KP_ANG_BASE_LEJOS = 2.0
KD_ANG_BASE_LEJOS = 0.2

KP_DIST_BASE_CERCA = 2.8
KD_DIST_BASE_CERCA = 0.8
KP_ANG_BASE_CERCA = 3.0
KD_ANG_BASE_CERCA = 0.5

KI_ANG_BASE = 0.01
MAX_INTEGRAL = 50.0

V_MAX_LINEAL_LEJOS = 180
V_MAX_LINEAL_CERCA = 255
V_MIN_LINEAL = 60
V_CERCA = 180
DIST_CERCA = 15

ALIGN_LOCK_DIST = 30
ALIGN_LOCK_HYSTERESIS = 1.5
OBSTACLE_CRITICAL_DIST = 35
GOAL_ALIGN_DIST = 60

MODO_GOL = False
PAUSA = True
PORTERIA_IZQUIERDA = True
ARUCO_ANGLE_OFFSET = 355

# Predictor robots
MAX_LOST_FRAMES = 8
MAX_LOST_FRAMES_RECOVERY = 15      # Umbral para activar modo recuperación

# Pelota: predicción solo en emergencia
FRAMES_ANTES_DE_PREDECIR = 15
MAX_PREDICT_FRAMES = 10

# =============================================================================
# FUNCIONES DE CARGA DE CONFIGURACIÓN
# =============================================================================
def load_color_config_hsv(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            lower = np.array(data['naranja']['lower'], dtype=np.uint8)
            upper = np.array(data['naranja']['upper'], dtype=np.uint8)
            return {'naranja': (lower, upper)}
    except Exception as e:
        print(f"Error crítico cargando JSON de colores HSV: {e}")
        sys.exit(1)

def load_field_config(file_path):
    global points
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                points = data.get('puntos_cancha', [])
                print("Configuración de cancha cargada automáticamente.")
        except Exception as e:
            print(f"Error leyendo JSON de cancha: {e}")
    else:
        print("No hay configuración de cancha. Por favor, haz clic en las 4 esquinas.")

def cargar_configuracion_equipo(file_path):
    """
    Lee el archivo JSON de equipo y retorna los datos completos.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Configuración de equipo cargada desde {file_path}")
            return data
    except Exception as e:
        print(f"Error cargando JSON de equipo: {e}")
        sys.exit(1)

def aplicar_equipo(data_equipo, equipo_actual):
    """
    Dado el diccionario completo del JSON y el nombre del equipo actual ('amarillo' o 'azul'),
    retorna el mapping de aliados (ID -> rol) y la lista de IDs rivales.
    """
    equipo_guardado = data_equipo.get('equipo_seleccionado', 'amarillo')
    aliados_dict = data_equipo.get('aliados', {})
    contrarios_dict = data_equipo.get('contrarios', {})

    if equipo_actual == equipo_guardado:
        # Usar configuración original
        mapping = {int(id_num): rol for rol, id_num in aliados_dict.items()}
        rivales = list(contrarios_dict.values())
        print(f"Equipo {equipo_actual.upper()} (original) activo.")
    else:
        # Intercambiar: nuestros robots son los contrarios del JSON, y viceversa
        mapping = {int(id_num): rol for rol, id_num in contrarios_dict.items()}
        rivales = list(aliados_dict.values())
        print(f"Equipo {equipo_actual.upper()} (invertido) activo.")

    return mapping, rivales

def load_camera_calibration(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
            dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float32)
            w, h = data['image_size']
            print(f"Calibración de cámara cargada: {file_path}")
            return camera_matrix, dist_coeffs, (w, h)
        except Exception as e:
            print(f"Error cargando calibración: {e}")
    else:
        print("No se encontró calibración de cámara.")
    return None, None, None

# =============================================================================
# VARIABLES GLOBALES
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- CARGAR CONFIGURACIÓN HSV ---
colors_hsv = load_color_config_hsv(os.path.join(script_dir, 'hsv_calibration.json'))

# --- CARGAR DATOS DEL EQUIPO (UNA SOLA VEZ) ---
DATOS_EQUIPO = cargar_configuracion_equipo(os.path.join(script_dir, 'equipo_config.json'))
EQUIPO_ACTUAL = DATOS_EQUIPO.get('equipo_seleccionado', 'amarillo')  # empezar con el guardado
role_mapping, rivales_ids = aplicar_equipo(DATOS_EQUIPO, EQUIPO_ACTUAL)

points = []
clahe_global = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Detector ArUco optimizado
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 30
aruco_params.cornerRefinementMinAccuracy = 0.1
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

camera_matrix, dist_coeffs, calib_img_size = load_camera_calibration(
    os.path.join(script_dir, 'camera_calibration.json')
)
use_undistort = camera_matrix is not None
map1_undist = None
map2_undist = None
last_frame_size = None

# Filtro Kalman SOLO para la pelota
kalman_ball = cv2.KalmanFilter(6, 2, 0)
kalman_ball.transitionMatrix = np.array([
    [1,0,1,0,0.5,0], [0,1,0,1,0,0.5], [0,0,1,0,1,0],
    [0,0,0,1,0,1], [0,0,0,0,1,0], [0,0,0,0,0,1]
], np.float32)
kalman_ball.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0]], np.float32)
kalman_ball.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
kalman_ball.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

last_errors = {'POR': 0.0, 'MED': 0.0, 'DEL': 0.0}
integral_errors = {'POR': 0.0, 'MED': 0.0, 'DEL': 0.0}
last_distances = {'POR': 0.0, 'MED': 0.0, 'DEL': 0.0}

KP_DIST = {rol: KP_DIST_BASE_LEJOS for rol in ['POR', 'MED', 'DEL']}
KD_DIST = {rol: KD_DIST_BASE_LEJOS for rol in ['POR', 'MED', 'DEL']}
KP_ANG = {rol: KP_ANG_BASE_LEJOS for rol in ['POR', 'MED', 'DEL']}
KD_ANG = {rol: KD_ANG_BASE_LEJOS for rol in ['POR', 'MED', 'DEL']}
KI_ANG = {rol: KI_ANG_BASE for rol in ['POR', 'MED', 'DEL']}

aligned_lock = {'POR': False, 'MED': False, 'DEL': False}
lock_angle = {'POR': 0.0, 'MED': 0.0, 'DEL': 0.0}
frames_sin_bola_lock = {'POR': 0, 'MED': 0, 'DEL': 0}

target_rosa = {'POR': None, 'MED': None, 'DEL': None}
modo_seleccion_rosa = 0

# Predictor simple para robots
robot_predictor = {
    'POR': {'last_pos': None, 'velocity': (0,0), 'lost_frames': 0, 'filtered_pos': None},
    'MED': {'last_pos': None, 'velocity': (0,0), 'lost_frames': 0, 'filtered_pos': None},
    'DEL': {'last_pos': None, 'velocity': (0,0), 'lost_frames': 0, 'filtered_pos': None}
}

# =============================================================================
# GEOMETRÍA Y VISIÓN
# =============================================================================
def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def click_event(event, x, y, flags, param):
    global points, target_rosa, modo_seleccion_rosa
    if event == cv2.EVENT_LBUTTONDOWN:
        if modo_seleccion_rosa > 0:
            if modo_seleccion_rosa == 1:
                target_rosa['POR'] = (x, y)
                print(f"Punto rosa POR: {target_rosa['POR']}")
                modo_seleccion_rosa = 2
            elif modo_seleccion_rosa == 2:
                target_rosa['MED'] = (x, y)
                print(f"Punto rosa MED: {target_rosa['MED']}")
                modo_seleccion_rosa = 3
            elif modo_seleccion_rosa == 3:
                target_rosa['DEL'] = (x, y)
                print(f"Punto rosa DEL: {target_rosa['DEL']}")
                modo_seleccion_rosa = 0
        elif len(points) < 4:
            points.append([x, y])

def warp_image(frame, src_pts):
    sorted_pts = order_points_clockwise(src_pts)
    ul, ur, lr, ll = sorted_pts
    widthA = np.linalg.norm(lr - ll); widthB = np.linalg.norm(ur - ul)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(ur - lr); heightB = np.linalg.norm(ul - ll)
    maxHeight = int(max(heightA, heightB))
    src = np.array([ul, ur, lr, ll], dtype="float32")
    dst = np.array([[0,0], [maxWidth-1,0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return warped, maxWidth, maxHeight

def detect_ball_hsv(hsv_image):
    lower, upper = colors_hsv['naranja']
    mask = cv2.inRange(hsv_image, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 3:
            return (int(x), int(y))
    return None

def identify_team_aruco(gray_frame):
    corners, ids, rejected = aruco_detector.detectMarkers(gray_frame)
    robots_found = {}
    posiciones_rivales = []
    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            if marker_id in role_mapping:
                rol = role_mapping[marker_id]
                front_x = (c[0][0] + c[1][0]) / 2.0
                front_y = (c[0][1] + c[1][1]) / 2.0
                vector_x = front_x - cx
                vector_y = front_y - cy
                angle_deg = (np.degrees(np.arctan2(vector_y, vector_x)) + 360) % 360
                angle_deg = (angle_deg + ARUCO_ANGLE_OFFSET) % 360
                robots_found[rol] = {'pos': (cx, cy), 'angle': angle_deg}
            elif marker_id in rivales_ids:
                posiciones_rivales.append((cx, cy))
    return robots_found, posiciones_rivales

# =============================================================================
# EVASIÓN Y GEOCERCA
# =============================================================================
def aplicar_repulsion(mi_pos, mi_target, obstaculos, dist_to_target=0):
    tx, ty = mi_target
    rx, ry = mi_pos
    factor_atenuacion = max(0.0, min(1.0, dist_to_target / 120.0)) if dist_to_target > 0 else 1.0
    for ox, oy in obstaculos:
        dist = np.hypot(ox - rx, oy - ry)
        if 0 < dist < RADIO_COLISION:
            dx = rx - ox
            dy = ry - oy
            mag = np.hypot(dx, dy) + 1e-8
            penetracion = RADIO_COLISION - dist
            fuerza = (penetracion / RADIO_COLISION) * FUERZA_REPULSION_MAX * factor_atenuacion
            fuerza = min(fuerza, 30)
            tx += (dx / mag) * fuerza
            ty += (dy / mag) * fuerza
    return int(tx), int(ty)

def aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h):
    """
    Aplica una fuerza de repulsión que aleja al robot de los bordes de la cancha.
    """
    fuerza_total_x = 0
    fuerza_total_y = 0

    # Borde izquierdo
    if rx < BORDE_DIST_SEGURIDAD:
        fuerza = (1.0 - rx / BORDE_DIST_SEGURIDAD) * FUERZA_REPULSION_BORDE
        fuerza_total_x += fuerza
    # Borde derecho
    if rx > frame_w - BORDE_DIST_SEGURIDAD:
        fuerza = (1.0 - (frame_w - rx) / BORDE_DIST_SEGURIDAD) * FUERZA_REPULSION_BORDE
        fuerza_total_x -= fuerza
    # Borde superior
    if ry < BORDE_DIST_SEGURIDAD:
        fuerza = (1.0 - ry / BORDE_DIST_SEGURIDAD) * FUERZA_REPULSION_BORDE
        fuerza_total_y += fuerza
    # Borde inferior
    if ry > frame_h - BORDE_DIST_SEGURIDAD:
        fuerza = (1.0 - (frame_h - ry) / BORDE_DIST_SEGURIDAD) * FUERZA_REPULSION_BORDE
        fuerza_total_y -= fuerza

    tx += fuerza_total_x
    ty += fuerza_total_y

    tx = max(0, min(frame_w, int(tx)))
    ty = max(0, min(frame_h, int(ty)))
    return tx, ty

def aplicar_geocerca_suave(tx, ty, area_x_min, area_x_max, area_y_min, area_y_max, porteria_izquierda):
    if porteria_izquierda:
        if tx >= (area_x_max + MARGEN_AREA) or ty <= (area_y_min - MARGEN_AREA) or ty >= (area_y_max + MARGEN_AREA):
            return int(tx), int(ty)
        nuevo_x = area_x_max + MARGEN_AREA + (tx - (area_x_max + MARGEN_AREA)) * 0.2
        return int(nuevo_x), int(ty)
    else:
        if tx <= (area_x_min - MARGEN_AREA) or ty <= (area_y_min - MARGEN_AREA) or ty >= (area_y_max + MARGEN_AREA):
            return int(tx), int(ty)
        nuevo_x = area_x_min - MARGEN_AREA - (tx - (area_x_min - MARGEN_AREA)) * 0.2
        return int(nuevo_x), int(ty)

# =============================================================================
# AUTO-TUNE DE GANANCIAS
# =============================================================================
def auto_tune_gains_lineal(role, distancia, d_dist):
    global KP_DIST, KD_DIST
    if distancia > 200:
        factor = 0.0
    elif distancia < 50:
        factor = 1.0
    else:
        factor = (200 - distancia) / 150.0

    kp = KP_DIST_BASE_LEJOS + factor * (KP_DIST_BASE_CERCA - KP_DIST_BASE_LEJOS)
    kd = KD_DIST_BASE_LEJOS + factor * (KD_DIST_BASE_CERCA - KD_DIST_BASE_LEJOS)

    if d_dist < -15:
        kp *= 0.8
        kd *= 1.3
    elif d_dist > 15:
        kp *= 1.2

    KP_DIST[role] = kp
    KD_DIST[role] = kd

def auto_tune_gains_angular(role, error_angular, d_error):
    global KP_ANG, KD_ANG, KI_ANG
    abs_err = abs(error_angular)

    if abs_err > 45:
        factor_base = 1.0
    elif abs_err > 20:
        factor_base = 0.7
    elif abs_err > 10:
        factor_base = 0.4
    else:
        factor_base = 0.2

    kp = KP_ANG_BASE_LEJOS + factor_base * (KP_ANG_BASE_CERCA - KP_ANG_BASE_LEJOS)
    kd = KD_ANG_BASE_LEJOS + factor_base * (KD_ANG_BASE_CERCA - KD_ANG_BASE_LEJOS)
    ki = KI_ANG_BASE * (0.5 + factor_base * 0.5)

    if abs(d_error) > 15:
        kp *= 0.7
        kd *= 1.3

    KP_ANG[role] = kp
    KD_ANG[role] = kd
    KI_ANG[role] = ki

# =============================================================================
# CONTROL SIMULTÁNEO (PD lineal + PID angular)
# =============================================================================
def control_pid_simultaneo(role, error_angular, distancia, flag, lock_active=False):
    global last_errors, integral_errors, last_distances

    if flag == 0:
        last_errors[role] = 0
        integral_errors[role] = 0
        last_distances[role] = distancia
        return 0, 0, 0, 0

    if flag == 2:
        if error_angular > 0:
            return 255, -255, 0, 255
        else:
            return -255, 255, 0, 255

    error_efectivo = error_angular
    d_error = error_efectivo - last_errors[role]
    if d_error > 180: d_error -= 360
    elif d_error < -180: d_error += 360

    d_dist = distancia - last_distances[role]
    auto_tune_gains_lineal(role, distancia, d_dist)
    auto_tune_gains_angular(role, error_efectivo, d_error)

    V = int(KP_DIST[role] * distancia + KD_DIST[role] * d_dist)
    if distancia > 200:
        v_max = V_MAX_LINEAL_LEJOS
    elif distancia < 50:
        v_max = V_MAX_LINEAL_CERCA
    else:
        v_max = V_MAX_LINEAL_LEJOS + (V_MAX_LINEAL_CERCA - V_MAX_LINEAL_LEJOS) * (200 - distancia) / 150.0
    V = max(V_MIN_LINEAL, min(int(v_max), V))

    P = error_efectivo
    D = d_error
    if abs(error_efectivo) < 60:
        integral_errors[role] += error_efectivo
        integral_errors[role] = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, integral_errors[role]))
    else:
        integral_errors[role] = 0
    I = integral_errors[role]

    W = int(KP_ANG[role] * P + KD_ANG[role] * D + KI_ANG[role] * I)

    if lock_active:
        W = 0

    last_errors[role] = error_efectivo
    last_distances[role] = distancia

    pwmL = V + W
    pwmR = V - W

    max_pwm = max(abs(pwmL), abs(pwmR))
    if max_pwm > 255:
        factor = 255.0 / max_pwm
        pwmL = int(pwmL * factor)
        pwmR = int(pwmR * factor)
        V = int(V * factor)
        W = int(W * factor)

    return pwmL, pwmR, V, W

def control_giro_violento(turn):
    if turn > 0:
        return 255, -255
    else:
        return -255, 255

# =============================================================================
# LÓGICA DE JUEGO (CON REPULSIÓN DE BORDES Y RECUPERACIÓN)
# =============================================================================
def calcular_logica(robots, target_ball, opp_goal, frame_w, frame_h, pos_rivales, porteria_izquierda):
    global MODO_GOL, aligned_lock, lock_angle, frames_sin_bola_lock
    bx, by = target_ball
    ox, oy = opp_goal

    if porteria_izquierda:
        my_goal_x, my_goal_y = 0, frame_h // 2
        area_x_min, area_x_max = 40, 95
        opp_goal_x, opp_goal_y = frame_w, frame_h // 2
    else:
        my_goal_x, my_goal_y = frame_w, frame_h // 2
        area_x_min, area_x_max = frame_w - 85, frame_w - 35
        opp_goal_x, opp_goal_y = 0, frame_h // 2

    area_y_min, area_y_max = frame_h // 4, (3 * frame_h) // 4
    area = (area_x_min, area_y_min, area_x_max, area_y_max, my_goal_x, my_goal_y)

    if (np.hypot(bx - opp_goal_x, by - opp_goal_y) <= 15) or (np.hypot(bx - my_goal_x, by - my_goal_y) <= 15):
        MODO_GOL = True
    if MODO_GOL:
        return "<0,0|0,0|0,0>\n", {}, area, {}, {'POR': (0,0), 'MED': (0,0), 'DEL': (0,0)}

    for rol in ['POR', 'MED', 'DEL']:
        if rol in robots:
            frames_sin_bola_lock[rol] = 0
        else:
            frames_sin_bola_lock[rol] += 1

    dist_MED = float('inf')
    dist_DEL = float('inf')
    if 'MED' in robots:
        mx, my = robots['MED']['pos']
        dist_MED = np.hypot(bx - mx, by - my)
    if 'DEL' in robots:
        dx, dy = robots['DEL']['pos']
        dist_DEL = np.hypot(bx - dx, by - dy)

    if 'DEL' in robots and 'MED' in robots:
        atacante_activo = 'DEL' if dist_DEL < dist_MED * PRIORIDAD_DELANTERO_THRESHOLD else 'MED'
    else:
        atacante_activo = 'DEL' if 'DEL' in robots else 'MED' if 'MED' in robots else None

    apoyo = None
    if atacante_activo:
        if atacante_activo == 'DEL' and 'MED' in robots:
            apoyo = 'MED'
        elif atacante_activo == 'MED' and 'DEL' in robots:
            apoyo = 'DEL'

    if atacante_activo is None:
        return "<0,0|0,0|0,0>\n", {}, area, {}, {'MED': (0,0), 'DEL': (0,0)}

    todas_posiciones = [data['pos'] for role, data in robots.items()]
    comandos_pwm = {'POR': "0,0", 'MED': "0,0", 'DEL': "0,0"}
    comandos_raw = {'POR': (0,0), 'MED': (0,0), 'DEL': (0,0)}
    debug_targets = {}
    debug_info = {}

    for role, data in robots.items():
        rx, ry = data['pos']
        ang = data['angle']
        dist_bola = np.hypot(bx - rx, by - ry)
        aliados = [p for p in todas_posiciones if p != (rx, ry)]
        obstaculos = aliados + pos_rivales

        # --- VERIFICAR MODO RECUPERACIÓN ---
        lost_frames = robot_predictor[role]['lost_frames']
        if lost_frames > MAX_LOST_FRAMES_RECOVERY:
            turn = 20 if (time.time() * 2) % 2 < 1 else -20
            pwmL, pwmR, V_actual, W_actual = control_pid_simultaneo(role, turn, 0, 1, lock_active=False)
            comandos_pwm[role] = f"{pwmL},{pwmR}"
            comandos_raw[role] = (pwmL, pwmR)
            debug_targets[role] = (rx, ry)
            debug_info[role] = {'error': turn, 'V': V_actual, 'W': W_actual, 'dist': 0,
                                'locked': False, 'modo': 'RECUPERACION'}
            continue

        modo_ataque = (role == atacante_activo) and (dist_bola < DIST_ACTIVACION_ATAQUE)

        if modo_ataque:
            ang_to_ball = (np.degrees(np.arctan2(by - ry, bx - rx)) + 360) % 360
            error_ang = ((ang_to_ball - ang + 180) % 360) - 180
            if not aligned_lock[role]:
                if (abs(error_ang) <= 20 and
                    dist_bola < ALIGN_LOCK_DIST and
                    frames_sin_bola_lock[role] == 0):
                    aligned_lock[role] = True
                    lock_angle[role] = ang
            else:
                if (dist_bola > ALIGN_LOCK_DIST * ALIGN_LOCK_HYSTERESIS or
                    frames_sin_bola_lock[role] > 5):
                    aligned_lock[role] = False
        else:
            aligned_lock[role] = False

        # --- Lógica por rol (con repulsión de bordes) ---
        if role == 'POR':
            if porteria_izquierda:
                target_x = 55
                target_y = my_goal_y
                area_x_lim = area_x_max
            else:
                target_x = frame_w - 45
                target_y = my_goal_y
                area_x_lim = area_x_min

            fuera_de_area = (rx > area_x_lim if porteria_izquierda else rx < area_x_lim) or \
                            ry < area_y_min or ry > area_y_max

            if fuera_de_area:
                tx, ty = target_x, target_y
                tx, ty = aplicar_repulsion((rx, ry), (tx, ty), obstaculos)
                tx, ty = aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h)
                ang_deseado = (np.degrees(np.arctan2(ty - ry, tx - rx)) + 360) % 360
                turn = ((ang_deseado - ang + 180) % 360) - 180
                distancia_control = np.hypot(tx - rx, ty - ry)
                flag = 1 if distancia_control > 15 else 0
                pwmL, pwmR, V_actual, W_actual = control_pid_simultaneo(role, turn, distancia_control, flag)
                comandos_pwm[role] = f"{pwmL},{pwmR}"
                comandos_raw[role] = (pwmL, pwmR)
                debug_targets['POR'] = (tx, ty)
                debug_info['POR'] = {'error': turn, 'V': V_actual, 'W': W_actual, 'dist': distancia_control, 'modo': 'PORTERO_VUELVE'}
                continue

            target_y = max(area_y_min + 20, min(by, area_y_max - 20))
            if dist_bola < CLEARANCE_DIST:
                flag = 2
            else:
                flag = 1
                if abs(rx - target_x) < 15 and abs(ry - target_y) < 15:
                    flag = 0

            if flag == 0 or flag == 2:
                ang_deseado = (np.degrees(np.arctan2(by - ry, bx - rx)) + 360) % 360
            else:
                ang_deseado = (np.degrees(np.arctan2(target_y - ry, target_x - rx)) + 360) % 360

            turn = ((ang_deseado - ang + 180) % 360) - 180
            if flag == 2:
                pwmL, pwmR = control_giro_violento(turn)
                V_actual = 255
                W_actual = 255
            else:
                tx_temp, ty_temp = target_x, target_y
                tx_temp, ty_temp = aplicar_repulsion_bordes(tx_temp, ty_temp, rx, ry, frame_w, frame_h)
                if flag != 0:
                    ang_deseado = (np.degrees(np.arctan2(ty_temp - ry, tx_temp - rx)) + 360) % 360
                    turn = ((ang_deseado - ang + 180) % 360) - 180
                pwmL, pwmR, V_actual, W_actual = control_pid_simultaneo(role, turn, dist_bola, flag)

            comandos_pwm[role] = f"{pwmL},{pwmR}"
            comandos_raw[role] = (pwmL, pwmR)
            debug_targets['POR'] = (target_x, target_y)
            debug_info['POR'] = {'error': turn, 'V': V_actual, 'W': W_actual, 'dist': dist_bola, 'modo': 'PORTERO'}

        elif role == atacante_activo:
            dist_to_goal = np.hypot(opp_goal[0] - rx, opp_goal[1] - ry)
            bloqueo_seguro = True
            if aligned_lock[role]:
                ang_rad = np.radians(lock_angle[role])
                dir_x = np.cos(ang_rad)
                dir_y = np.sin(ang_rad)
                for ox, oy in obstaculos:
                    dx = ox - rx
                    dy = oy - ry
                    dist_obs = np.hypot(dx, dy)
                    if dist_obs < OBSTACLE_CRITICAL_DIST:
                        dot = (dx * dir_x + dy * dir_y) / (dist_obs + 1e-8)
                        if dot > 0.5:
                            bloqueo_seguro = False
                            break

            if aligned_lock[role] and bloqueo_seguro and dist_to_goal > GOAL_ALIGN_DIST:
                tx, ty = opp_goal
                tx, ty = aplicar_repulsion((rx, ry), (tx, ty), obstaculos)
                tx, ty = aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h)
                if np.hypot(tx - rx, ty - ry) > 1e-3:
                    ang_deseado = (np.degrees(np.arctan2(ty - ry, tx - rx)) + 360) % 360
                else:
                    ang_deseado = lock_angle[role]
                distancia_control = dist_to_goal
                flag = 1
                modo_str = "ATAQUE_BLOQ_PORTERIA"
            else:
                if aligned_lock[role]:
                    aligned_lock[role] = False
                if dist_to_goal < GOAL_ALIGN_DIST:
                    tx, ty = opp_goal
                    tx, ty = aplicar_repulsion((rx, ry), (tx, ty), obstaculos)
                    tx, ty = aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h)
                    if np.hypot(tx - rx, ty - ry) > 1e-3:
                        ang_deseado = (np.degrees(np.arctan2(ty - ry, tx - rx)) + 360) % 360
                    else:
                        ang_deseado = ang
                    distancia_control = dist_to_goal
                    flag = 1
                    modo_str = "ATAQUE_CERCA_ARCO"
                else:
                    dx_b = bx - opp_goal[0]
                    dy_b = by - opp_goal[1]
                    mag = np.hypot(dx_b, dy_b) + 1e-8
                    tx = bx + (dx_b / mag) * DISTANCIA_DETRAS_PELOTA
                    ty = by + (dy_b / mag) * DISTANCIA_DETRAS_PELOTA
                    if np.hypot(tx - rx, ty - ry) < 40:
                        tx, ty = bx, by
                    tx, ty = aplicar_geocerca_suave(tx, ty, area_x_min, area_x_max, area_y_min, area_y_max, porteria_izquierda)
                    tx, ty = aplicar_repulsion((rx, ry), (tx, ty), obstaculos)
                    tx, ty = aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h)
                    if np.hypot(tx - rx, ty - ry) > 1e-3:
                        ang_deseado = (np.degrees(np.arctan2(ty - ry, tx - rx)) + 360) % 360
                    else:
                        ang_deseado = ang
                    distancia_control = dist_bola
                    flag = 1
                    modo_str = "ATAQUE_DETRAS"

            turn = ((ang_deseado - ang + 180) % 360) - 180
            usar_lock = (aligned_lock[role] and bloqueo_seguro)
            pwmL, pwmR, V_actual, W_actual = control_pid_simultaneo(role, turn, distancia_control, flag, lock_active=usar_lock)
            comandos_pwm[role] = f"{pwmL},{pwmR}"
            comandos_raw[role] = (pwmL, pwmR)
            debug_targets[role] = (int(tx), int(ty))
            debug_info[role] = {'error': turn, 'V': V_actual, 'W': W_actual, 'dist': distancia_control, 'locked': aligned_lock[role], 'modo': modo_str}

        elif role == apoyo:
            if target_rosa.get(role) is not None and dist_bola >= DIST_ACTIVACION_ATAQUE:
                tx, ty = target_rosa[role]
                distancia_control = np.hypot(tx - rx, ty - ry)
                flag = 1 if distancia_control > 15 else 0
                tx, ty = aplicar_repulsion((rx, ry), (tx, ty), obstaculos, dist_to_target=distancia_control)
                tx, ty = aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h)
                if np.hypot(tx - rx, ty - ry) > 1e-3:
                    ang_deseado = (np.degrees(np.arctan2(ty - ry, tx - rx)) + 360) % 360
                else:
                    ang_deseado = ang
                turn = ((ang_deseado - ang + 180) % 360) - 180
                pwmL, pwmR, V_actual, W_actual = control_pid_simultaneo(role, turn, distancia_control, flag)
                comandos_pwm[role] = f"{pwmL},{pwmR}"
                comandos_raw[role] = (pwmL, pwmR)
                debug_targets[role] = (int(tx), int(ty))
                debug_info[role] = {'error': turn, 'V': V_actual, 'W': W_actual, 'dist': distancia_control, 'locked': False, 'modo': 'APOYO_PUNTO_ROSA'}
                continue

            if dist_bola < APOYO_INTERCEPT_DIST:
                dx_b = bx - opp_goal[0]
                dy_b = by - opp_goal[1]
                mag = np.hypot(dx_b, dy_b) + 1e-8
                tx = bx + (dx_b / mag) * 10
                ty = by + (dy_b / mag) * 10
                flag = 1
                modo_str = "APOYO_INTERCEPT"
            else:
                midfield_x = frame_w // 2
                if porteria_izquierda:
                    defensive_x = int(frame_w * APOYO_DEFENSIVE_X_FACTOR)
                else:
                    defensive_x = frame_w - int(frame_w * APOYO_DEFENSIVE_X_FACTOR)
                target_x = midfield_x if bx > midfield_x else defensive_x
                target_y = by
                tx, ty = target_x, target_y
                dist_to_target = np.hypot(tx - rx, ty - ry)
                flag = 1 if dist_to_target > 40 else 0
                modo_str = "APOYO_POSICION"

            if flag != 0:
                tx, ty = aplicar_geocerca_suave(tx, ty, area_x_min, area_x_max, area_y_min, area_y_max, porteria_izquierda)
                tx, ty = aplicar_repulsion((rx, ry), (tx, ty), obstaculos)
                tx, ty = aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h)

            if flag != 0:
                if np.hypot(tx - rx, ty - ry) > 1e-3:
                    ang_deseado = (np.degrees(np.arctan2(ty - ry, tx - rx)) + 360) % 360
                else:
                    ang_deseado = ang
            else:
                ang_deseado = (np.degrees(np.arctan2(by - ry, bx - rx)) + 360) % 360

            turn = ((ang_deseado - ang + 180) % 360) - 180
            distancia_control = dist_bola if flag == 0 else np.hypot(tx - rx, ty - ry)
            pwmL, pwmR, V_actual, W_actual = control_pid_simultaneo(role, turn, distancia_control, flag)
            comandos_pwm[role] = f"{pwmL},{pwmR}"
            comandos_raw[role] = (pwmL, pwmR)
            debug_targets[role] = (int(tx), int(ty))
            debug_info[role] = {'error': turn, 'V': V_actual, 'W': W_actual, 'dist': distancia_control, 'locked': False, 'modo': modo_str}

        else:
            comandos_pwm[role] = "0,0"
            comandos_raw[role] = (0,0)
            debug_info[role] = {'modo': 'INACTIVO'}

    trama = f"<{comandos_pwm['POR']}|{comandos_pwm['MED']}|{comandos_pwm['DEL']}>\n"
    return trama, debug_targets, area, debug_info, comandos_raw

# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================
def main():
    global points, MODO_GOL, PAUSA, PORTERIA_IZQUIERDA, ARUCO_ANGLE_OFFSET
    global use_undistort, camera_matrix, dist_coeffs, map1_undist, map2_undist, last_frame_size
    global modo_seleccion_rosa, target_rosa
    global robot_predictor, EQUIPO_ACTUAL, role_mapping, rivales_ids

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='COM9')
    args = parser.parse_args()

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('Campo Robofutbol')
    cv2.setMouseCallback('Campo Robofutbol', click_event)

    try:
        esp32 = serial.Serial(args.port, 115200, timeout=0.01)
        esp32.setDTR(False); esp32.setRTS(False)
        time.sleep(2)
        print("ESP32 Conectado.")
    except:
        esp32 = None
        print("Modo Simulación (sin ESP32).")

    memoria_robots = {rol: {'pos': None, 'angle': 0} for rol in ['POR', 'MED', 'DEL']}
    frames_sin_bola = 0
    load_field_config(os.path.join(script_dir, 'cancha_config.json'))

    last_serial_time = time.time()
    last_trama = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        current_time = time.time()

        if use_undistort:
            h, w = frame.shape[:2]
            if last_frame_size != (w, h):
                map1_undist, map2_undist = cv2.fisheye.initUndistortRectifyMap(
                    camera_matrix, dist_coeffs, np.eye(3), camera_matrix, (w, h), cv2.CV_32FC1
                )
                last_frame_size = (w, h)
            frame = cv2.remap(frame, map1_undist, map2_undist, cv2.INTER_LINEAR)

        if len(points) == 4:
            try:
                warped, w, h = warp_image(frame, points)
            except:
                points.clear()
                continue

            hsv_image = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = clahe_global.apply(gray)
            warped_display = warped.copy()

            ball_pos = detect_ball_hsv(hsv_image)
            pred_ball = kalman_ball.predict()

            if ball_pos:
                target_ball = ball_pos
                frames_sin_bola = 0
                kalman_ball.correct(np.array([[ball_pos[0]], [ball_pos[1]]], np.float32))
            else:
                frames_sin_bola += 1
                if frames_sin_bola >= FRAMES_ANTES_DE_PREDECIR and frames_sin_bola < FRAMES_ANTES_DE_PREDECIR + MAX_PREDICT_FRAMES:
                    target_ball = (int(pred_ball[0][0]), int(pred_ball[1][0]))
                else:
                    if frames_sin_bola >= FRAMES_ANTES_DE_PREDECIR + MAX_PREDICT_FRAMES:
                        kalman_ball.statePost = np.zeros((6,1), np.float32)
                        kalman_ball.errorCovPost = np.eye(6, dtype=np.float32) * 1
                    target_ball = None

            detectados, rivales_en_cancha = identify_team_aruco(gray)
            robots_activos = {}

            for rol in ['POR', 'MED', 'DEL']:
                pred = robot_predictor[rol]

                if rol in detectados:
                    x, y = detectados[rol]['pos']
                    ang = detectados[rol]['angle']

                    if pred['last_pos'] is not None:
                        vx = x - pred['last_pos'][0]
                        vy = y - pred['last_pos'][1]
                        alpha = 0.7
                        pred['velocity'] = (alpha * vx + (1-alpha)*pred['velocity'][0],
                                            alpha * vy + (1-alpha)*pred['velocity'][1])
                    pred['last_pos'] = (x, y)
                    pred['lost_frames'] = 0
                    pred['filtered_pos'] = (x, y)

                    memoria_robots[rol]['pos'] = (x, y)
                    memoria_robots[rol]['angle'] = ang
                    robots_activos[rol] = memoria_robots[rol]

                else:
                    pred['lost_frames'] += 1
                    if pred['last_pos'] is not None and pred['lost_frames'] <= MAX_LOST_FRAMES:
                        vx, vy = pred['velocity']
                        px = pred['last_pos'][0] + vx * pred['lost_frames']
                        py = pred['last_pos'][1] + vy * pred['lost_frames']
                        px = max(0, min(w, int(px)))
                        py = max(0, min(h, int(py)))
                        pred['filtered_pos'] = (px, py)

                        memoria_robots[rol]['pos'] = (px, py)
                        robots_activos[rol] = memoria_robots[rol]
                    else:
                        memoria_robots[rol]['pos'] = None
                        pred['filtered_pos'] = None
                        pred['velocity'] = (0,0)

                if pred['filtered_pos'] is not None:
                    rx, ry = pred['filtered_pos']
                    color = (255,100,0) if pred['lost_frames'] == 0 else (0,255,255)
                    cv2.circle(warped_display, (rx, ry), 20, color, 2)
                    cv2.putText(warped_display, rol, (rx-15, ry-25), 0, 0.6, (255,255,255), 2)
                    if memoria_robots[rol]['angle'] is not None:
                        ang_rad = np.radians(memoria_robots[rol]['angle'])
                        lx = int(rx + 30 * np.cos(ang_rad))
                        ly = int(ry + 30 * np.sin(ang_rad))
                        cv2.arrowedLine(warped_display, (rx, ry), (lx, ly), (0,255,255), 2, tipLength=0.3)

            for rx, ry in rivales_en_cancha:
                cv2.circle(warped_display, (rx, ry), 22, (0,0,255), 2)
                cv2.putText(warped_display, "RIVAL", (rx-15, ry-25), 0, 0.4, (0,0,255), 1)

            for rol, pt in target_rosa.items():
                if pt is not None:
                    cv2.circle(warped_display, pt, 10, (255, 0, 255), -1)
                    cv2.putText(warped_display, rol, (pt[0]-15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if target_ball and robots_activos:
                opp_goal = (w, h // 2) if PORTERIA_IZQUIERDA else (0, h // 2)
                trama, debug_targets, area, debug_info, _ = calcular_logica(
                    robots_activos, target_ball, opp_goal, w, h, rivales_en_cancha, PORTERIA_IZQUIERDA
                )

                ax_min, ay_min, ax_max, ay_max, goal_x, goal_y = area
                cv2.circle(warped_display, opp_goal, 20, (0,0,255), 3)
                cv2.line(warped_display, (opp_goal[0]-20, opp_goal[1]), (opp_goal[0]+20, opp_goal[1]), (0,0,255), 2)
                cv2.line(warped_display, (opp_goal[0], opp_goal[1]-20), (opp_goal[0], opp_goal[1]+20), (0,0,255), 2)
                cv2.rectangle(warped_display, (ax_min, ay_min), (ax_max, ay_max), (255,255,0), 2)

                for rol, data in robots_activos.items():
                    rx, ry = data['pos']
                    cv2.circle(warped_display, (rx, ry), RADIO_COLISION, (0, 255, 0), 1)
                    if rol in debug_info:
                        err = debug_info[rol]['error']
                        vel = debug_info[rol]['V']
                        w_vel = debug_info[rol].get('W', 0)
                        dist = debug_info[rol]['dist']
                        locked = debug_info[rol].get('locked', False)
                        modo = debug_info[rol].get('modo', '')
                        status = f"LOCK {modo}" if locked else modo
                        cv2.putText(warped_display, f"E:{int(err)} V:{vel} W:{w_vel} D:{int(dist)} {status}",
                                    (rx+20, ry-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0,255,0) if locked else (0,255,255), 1)
                    if rol in debug_targets:
                        tx, ty = debug_targets[rol]
                        cv2.circle(warped_display, (tx, ty), 8, (255,0,255), -1)
                        cv2.putText(warped_display, rol, (tx-15, ty-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                        cv2.line(warped_display, (rx, ry), (tx, ty), (255,0,255), 1)

                if MODO_GOL:
                    cv2.putText(warped_display, "GOOOOOL!", (w//2-150, h//2), 0, 2, (0,255,255), 5)

                cv2.putText(warped_display, f"Enviando: {trama.strip()}", (10,30), 0, 0.6, (0,255,255), 2)
                ball_x, ball_y = target_ball
                cv2.circle(warped_display, (ball_x, ball_y), 5, (0,165,255), -1)
                cv2.putText(warped_display, f"Pelota: X:{ball_x} Y:{ball_y}", (10,60), 0, 0.6, (0,255,0), 2)
            else:
                trama = "<0,0|0,0|0,0>\n"
                cv2.putText(warped_display, "Esperando pelota/robots", (10,30), 0, 0.6, (0,0,255), 2)

            if PAUSA:
                trama = "<0,0|0,0|0,0>\n"
                cv2.putText(warped_display, "PAUSA", (w-100, 50), 0, 1, (0,0,255), 3)

            cv2.putText(warped_display, f"Porteria: {'IZQ' if PORTERIA_IZQUIERDA else 'DER'}", (10, h-20), 0, 0.6, (0,255,255), 2)
            cv2.putText(warped_display, f"Offset ArUco: {ARUCO_ANGLE_OFFSET} (+/-)", (10, h-40), 0, 0.5, (200,200,200), 1)
            cv2.putText(warped_display, f"Equipo: {EQUIPO_ACTUAL.upper()} (tecla 'e' para cambiar)", (10, h-60), 0, 0.5, (200,200,200), 1)

            if modo_seleccion_rosa > 0:
                texto = f"Selecciona punto ROSA para: {'POR' if modo_seleccion_rosa==1 else 'MED' if modo_seleccion_rosa==2 else 'DEL'}"
                cv2.putText(warped_display, texto, (10, h-80), 0, 0.7, (255, 0, 255), 2)

            if esp32:
                tiempo_transcurrido = current_time - last_serial_time
                if (trama != last_trama and tiempo_transcurrido > 0.02) or (tiempo_transcurrido > 0.2):
                    esp32.write(trama.encode())
                    last_serial_time = current_time
                    last_trama = trama

            cv2.imshow('Campo Robofutbol', warped_display)

        else:
            cv2.putText(frame, "Clica 4 esquinas (sup izq, sup der, inf der, inf izq)", (10,30), 0, 0.7, (0,0,255), 2)
            for p in points:
                cv2.circle(frame, tuple(p), 5, (0,255,0), -1)
            cv2.imshow('Campo Robofutbol', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if esp32:
                esp32.write("<0,0|0,0|0,0>\n".encode())
            break
        elif key == ord('r'):
            points.clear()
        elif key == ord('g'):
            MODO_GOL = False
        elif key == ord('f'):
            PAUSA = not PAUSA
            if PAUSA and esp32:
                esp32.write("<0,0|0,0|0,0>\n".encode())
        elif key == ord('p'):
            PORTERIA_IZQUIERDA = not PORTERIA_IZQUIERDA
        elif key == ord('+') or key == ord('='):
            ARUCO_ANGLE_OFFSET = (ARUCO_ANGLE_OFFSET + 5) % 360
        elif key == ord('-') or key == ord('_'):
            ARUCO_ANGLE_OFFSET = (ARUCO_ANGLE_OFFSET - 5) % 360
        elif key == ord('t'):
            modo_seleccion_rosa = 1
        elif key == ord('e'):
            # Alternar equipo entre amarillo y azul
            EQUIPO_ACTUAL = 'azul' if EQUIPO_ACTUAL == 'amarillo' else 'amarillo'
            role_mapping, rivales_ids = aplicar_equipo(DATOS_EQUIPO, EQUIPO_ACTUAL)
            print(f"Equipo cambiado a: {EQUIPO_ACTUAL.upper()}")

    cap.release()
    if esp32:
        esp32.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()