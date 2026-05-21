# ======================================================================
# Plataforma desarrollada para utilizarse en el Torneo Zacatecano de
# Fútbol Robótico en la categoría VSSS, organizado por el Consejo
# Zacatecano de Ciencia, Tecnología e Innovación (COZCYT) y el Club
# de Robótica RobTai-UAZ
# Desarrollada por: M. en C. José Manuel Mercado Blanco
#
# Descripción general:
# Sistema de visión y control para tres robots de fútbol (VSSS).
# Incluye:
#   - Calibración de campo y colores.
#   - Identificación de robots propios y rivales mediante patrones de
#     colores.
#   - Estrategia de juego para el robot delantero (DEL), incluyendo
#     control PID, evitación de obstáculos y bordes.
#   - Espacio reservado para POR (portero) y MED (mediocampista) para
#     que los participantes añadan sus propias estrategias.
#   - Interfaz gráfica con ajustes de cámara, repulsión, y corrección
#     de distorsión radial.
# ======================================================================

import cv2
import numpy as np
import json
import os
import sys
import time
import serial
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# ============================================================
# NOMBRES DE LOS ROBOTS (modificables si se desea)
# ============================================================
robot_POR = "POR"   # Nombre del robot portero
robot_MED = "MED"   # Nombre del robot mediocampista
robot_DEL = "DEL"   # Nombre del robot delantero

# ============================================================
# CONFIGURACIONES GLOBALES
# ============================================================
# Distancia de seguridad para evitar colisiones (px)
CLEARANCE_DIST = 80
# Número de frames que se recuerda la posición de un robot perdido
MEMORY_FRAMES = 15
# Frames antes de congelar la posición estimada
MAX_LOST_FRAMES = 8
# Frames para activar el giro de búsqueda (modo recuperación)
MAX_LOST_FRAMES_RECOVERY = 15

# Comunicación serial
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 0.01

# Dimensiones reales de la cancha en cm (ancho, alto)
FIELD_ASPECT_RATIO = (150, 130)
# Resolución de salida de la imagen corregida (píxeles)
BASE_RESOLUTION = (750, 650)

# Márgenes iniciales para calibración HSV (mejorados)
DEFAULT_H_MARGIN = 12
DEFAULT_S_MARGIN = 70
DEFAULT_V_MARGIN = 70

# ----------------------------------------------------------
# MAPEO DE COLORES (cuadrados) -> ROL
# Combinación de dos colores de marcador → rol del robot
# ----------------------------------------------------------
ROLE_COLOR_PAIRS = {
    ('rosa', 'verde'): robot_DEL,
    ('azul', 'rosa'):  robot_POR,
    ('azul', 'verde'): robot_MED
}

# Colores para robots propios (BGR)
COLOR_POR_OWN = (0, 255, 0)
COLOR_MED_OWN = (0, 255, 0)
COLOR_DEL_OWN = (0, 255, 0)
# Color para robots rivales (todos con el mismo color)
COLOR_RIVAL = (0, 0, 255)

# Colores generales
COLOR_BALL = (0, 165, 255)          # Naranja para la pelota
COLOR_HUD_BG = (0, 0, 0)           # Fondo del HUD
COLOR_TARGET = (255, 0, 255)        # Rosa para punto de destino
COLOR_REPULSION = (0, 0, 255)       # Rojo para zonas de repulsión

# ----------------------------------------------------------
# CONSTANTES DEL CONTROLADOR PID
# ----------------------------------------------------------
# Constante proporcional: si se aumenta, el robot responde más rápido
# pero puede oscilar.
Kp_ang = 0.8
# Constante integral: reduce el error en estado estable, pero valores
# altos provocan sobreimpulso.
Ki_ang = 0.01
# Constante derivativa: suaviza la respuesta, valores altos pueden
# introducir ruido.
Kd_ang = 0.1

# Paso de tiempo para el control PID (asumiendo ~30 FPS)
DT = 1.0 / 30.0

# ============================================================
# FUNCIÓN DE CONTROL PID
# ============================================================
def control_pid_angular(error, dt, integral, prev_error):
    """
    Calcula la señal de control PID para el error angular.

    Parámetros:
        error (float): Error angular actual (grados).
        dt (float): Intervalo de tiempo desde la última actualización.
        integral (float): Valor acumulado del error.
        prev_error (float): Error anterior.

    Retorna:
        output (float): Señal de control.
        integral (float): Nuevo valor integral.
        prev_error (float): Error actual (para la siguiente llamada).
    """
    # Término integral con anti-windup simple (limitar acumulación)
    integral = integral + error * dt
    integral = max(min(integral, 50), -50)   # saturación
    derivada = (error - prev_error) / dt if dt > 0 else 0
    output = Kp_ang * error + Ki_ang * integral + Kd_ang * derivada
    return output, integral, error

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def load_json(file_path):
    """Carga un archivo JSON y devuelve el diccionario."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando {file_path}: {e}")
        return None

def save_json(file_path, data):
    """Guarda un diccionario en un archivo JSON."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error guardando {file_path}: {e}")
        return False

def order_points_clockwise(pts):
    """
    Ordena cuatro puntos en sentido horario empezando por el
    superior-izquierdo (UL → UR → LR → LL).

    Parámetros:
        pts (list/tuple/ndarray): Cuatro puntos (x,y).

    Retorna:
        ndarray: Puntos ordenados con forma (4,2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # UL (menor suma x+y)
    rect[2] = pts[np.argmax(s)]   # LR (mayor suma)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # UR (menor diferencia y-x)
    rect[3] = pts[np.argmax(diff)] # LL (mayor diferencia)
    return rect

def init_kalman():
    """
    Inicializa un filtro de Kalman para el seguimiento de la pelota.
    Modelo de velocidad constante.
    """
    kf = cv2.KalmanFilter(6, 2, 0)
    kf.transitionMatrix = np.array([
        [1,0,1,0,0.5,0],
        [0,1,0,1,0,0.5],
        [0,0,1,0,1,0],
        [0,0,0,1,0,1],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0]], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf

def undistort_frame(frame, camera_matrix, dist_coeffs):
    """
    Corrige la distorsión radial de la imagen usando la matriz y los
    coeficientes de distorsión de la cámara.

    Parámetros:
        frame (ndarray): Imagen BGR.
        camera_matrix (ndarray): Matriz intrínseca de la cámara.
        dist_coeffs (ndarray): Coeficientes de distorsión.

    Retorna:
        ndarray: Imagen corregida.
    """
    if camera_matrix is not None and dist_coeffs is not None:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
        return undistorted
    return frame

def warp_image(frame, src_pts, output_size=BASE_RESOLUTION):
    """
    Aplica una transformación de perspectiva para obtener una vista
    cenital del campo. Los puntos src_pts se corresponden con las
    cuatro esquinas del campo en la imagen (UL, UR, LR, LL).

    Parámetros:
        frame (ndarray): Imagen original.
        src_pts (list): Cuatro puntos de las esquinas del campo.
        output_size (tuple): Tamaño (ancho, alto) de la imagen resultante.

    Retorna:
        warped (ndarray): Imagen rectificada.
        newWidth (int): Ancho de la imagen rectificada.
        newHeight (int): Alto de la imagen rectificada.
    """
    sorted_pts = order_points_clockwise(src_pts)
    ul, ur, lr, ll = sorted_pts   # UL: esquina superior izquierda, UR: superior derecha, etc.
    src = np.array([ul, ur, lr, ll], dtype="float32")
    newWidth, newHeight = output_size
    dst = np.array([
        [0, 0],
        [newWidth - 1, 0],
        [newWidth - 1, newHeight - 1],
        [0, newHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (newWidth, newHeight), flags=cv2.INTER_CUBIC)
    return warped, newWidth, newHeight

def detect_ball(hsv, ball_range):
    """
    Detecta la pelota dentro de la imagen HSV.

    Parámetros:
        hsv (ndarray): Imagen en espacio HSV.
        ball_range (tuple): (lower, upper) límites HSV.

    Retorna:
        tuple o None: (x, y) del centro de la pelota, o None si no se encuentra.
    """
    lower, upper = ball_range
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 3:
            return (int(x), int(y))
    return None

def get_rects(hsv, lower, upper, min_a=15, max_a=5000):
    """
    Obtiene los rectángulos rotados de las regiones que caen dentro
    del rango de color indicado.

    Parámetros:
        hsv (ndarray): Imagen HSV.
        lower (ndarray): Límite inferior HSV.
        upper (ndarray): Límite superior HSV.
        min_a (int): Área mínima de contorno.
        max_a (int): Área máxima de contorno.

    Retorna:
        list: Lista de rectángulos (cv2.minAreaRect).
    """
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.minAreaRect(c) for c in contours if min_a < cv2.contourArea(c) < max_a]

def get_closest_marker_center(base_center, markers, radius):
    """
    Encuentra el marcador más cercano al centro de la base del robot.

    Parámetros:
        base_center (tuple): (x, y) del centro de la base.
        markers (list): Lista de rectángulos de marcadores.
        radius (float): Radio máximo de búsqueda.

    Retorna:
        ndarray o None: Centro del marcador más cercano o None.
    """
    closest_center = None
    min_dist = radius
    for m in markers:
        (mx, my), _, _ = m
        dist = np.hypot(mx - base_center[0], my - base_center[1])
        if dist < min_dist:
            min_dist = dist
            closest_center = np.array([mx, my])
    return closest_center

# ============================================================
# FILTRO DE FALSOS MARCADORES
# ============================================================
def filter_out_base_markers(bases, markers_dict, min_dist=10):
    """
    Elimina de markers_dict cualquier marcador cuyo centro esté a menos
    de min_dist píxeles del centro de alguna base.

    Parámetros:
        bases (list): Lista de rectángulos de las bases de los equipos.
        markers_dict (dict): Diccionario con listas de marcadores por color.
        min_dist (int): Distancia mínima para considerar un marcador como
                        parte de la base.

    Retorna:
        dict: Diccionario de marcadores filtrado.
    """
    filtered = {}
    for color_name, marker_list in markers_dict.items():
        filtered[color_name] = []
        for m in marker_list:
            (mx, my), _, _ = m
            keep = True
            for base in bases:
                (bx, by), _, _ = base
                if np.hypot(mx - bx, my - by) < min_dist:
                    keep = False
                    break
            if keep:
                filtered[color_name].append(m)
    return filtered

# ============================================================
# IDENTIFICACIÓN DE ROBOTS
# ============================================================
def identify_team(hsv, colors, my_color):
    """
    Identifica los robots propios (con roles POR, MED, DEL) y los
    rivales (sin distinción de rol, solo como 'Rival').

    Parámetros:
        hsv (ndarray): Imagen HSV.
        colors (dict): Diccionario con rangos HSV para cada color.
        my_color (str): 'azul' o 'amarillo', color de nuestro equipo.

    Retorna:
        own (dict): Robots propios con roles y sus datos (pos, angle).
        rivals (list): Lista de diccionarios con 'pos' y 'angle' de cada rival.
    """
    # Rangos de las bases según el color elegido
    amarillo_range = colors['amarillo']
    azul_range = colors['azul']
    if my_color == 'azul':
        my_base_range = azul_range
        rival_base_range = amarillo_range
    else:
        my_base_range = amarillo_range
        rival_base_range = azul_range

    # Detectar bases
    my_bases = get_rects(hsv, my_base_range[0], my_base_range[1], 50, 5000)
    rival_bases = get_rects(hsv, rival_base_range[0], rival_base_range[1], 50, 5000)

    # Detectar marcadores de colores
    markers_p = get_rects(hsv, colors['rosa'][0], colors['rosa'][1], 15, 2000)
    markers_g = get_rects(hsv, colors['verde'][0], colors['verde'][1], 15, 2000)
    markers_b = get_rects(hsv, colors['azul'][0], colors['azul'][1], 15, 2000)

    available_markers = {
        'rosa': markers_p,
        'verde': markers_g,
        'azul': markers_b
    }

    # Filtrar marcadores que están sobre las bases
    filtered_my = filter_out_base_markers(my_bases, available_markers)
    filtered_rival = filter_out_base_markers(rival_bases, available_markers)

    # Robots propios: se asigna rol según combinación de colores
    own = {robot_POR: None, robot_MED: None, robot_DEL: None}

    used_markers = set()
    for base in my_bases:
        (bx, by), (bw, bh), _ = base
        base_center = np.array([bx, by])
        radius = max(bw, bh) * 1.5

        closest_of_color = {}
        for color_name, marker_list in filtered_my.items():
            candidate = None
            min_d = radius
            for i, m in enumerate(marker_list):
                (mx, my), _, _ = m
                d = np.hypot(mx - bx, my - by)
                if d < min_d:
                    min_d = d
                    candidate = (i, np.array([mx, my]))
            if candidate is not None:
                closest_of_color[color_name] = candidate

        found_colors = list(closest_of_color.keys())
        if len(found_colors) != 2:
            continue
        pair = tuple(sorted(found_colors))
        if pair not in ROLE_COLOR_PAIRS:
            continue
        role = ROLE_COLOR_PAIRS[pair]

        idx1 = closest_of_color[found_colors[0]][0]
        idx2 = closest_of_color[found_colors[1]][0]
        if (found_colors[0], idx1) in used_markers or (found_colors[1], idx2) in used_markers:
            continue
        used_markers.add((found_colors[0], idx1))
        used_markers.add((found_colors[1], idx2))

        pos1 = closest_of_color[found_colors[0]][1]
        pos2 = closest_of_color[found_colors[1]][1]
        mean_marker_pos = (pos1 + pos2) / 2.0
        front_vector = mean_marker_pos - base_center 
        angle_deg = (np.degrees(np.arctan2(front_vector[1], front_vector[0])) + 360) % 360

        own[role] = {'pos': (int(bx), int(by)), 'angle': angle_deg}

    # Robots rivales: solo posición y ángulo, sin rol específico
    rivals = []
    for base in rival_bases:
        (bx, by), (bw, bh), _ = base
        base_center = np.array([bx, by])
        radius = max(bw, bh) * 1.5

        # Buscar cualquier marcador cercano para calcular orientación
        closest_marker = None
        min_dist = radius
        for color_name, marker_list in filtered_rival.items():
            for m in marker_list:
                (mx, my), _, _ = m
                d = np.hypot(mx - bx, my - by)
                if d < min_dist:
                    min_dist = d
                    closest_marker = np.array([mx, my])
        if closest_marker is not None:
            front_vector = base_center - closest_marker
            angle = (np.degrees(np.arctan2(front_vector[1], front_vector[0])) + 360) % 360
            rivals.append({'pos': (int(bx), int(by)), 'angle': angle})
        else:
            # Si no se encuentra marcador, no se añade orientación
            pass

    return own, rivals

# ============================================================
# FUNCIONES DE REPULSIÓN Y PREDICTOR
# ============================================================
def aplicar_repulsion(mi_pos, target, obstaculos, radio_colision, fuerza_max, dist_to_target=0):
    """
    Desvía el punto objetivo para evitar colisiones con otros robots.

    Parámetros:
        mi_pos (tuple): Posición (x,y) del robot propio.
        target (tuple): Punto objetivo original (x,y).
        obstaculos (list): Lista de posiciones (x,y) de otros robots.
        radio_colision (int): Radio de la zona de repulsión.
        fuerza_max (int): Fuerza máxima de repulsión.
        dist_to_target (float): Distancia actual al objetivo (atenúa la fuerza).

    Retorna:
        tuple: Nuevo objetivo (x,y) modificado.
    """
    tx, ty = target
    rx, ry = mi_pos
    factor_atenuacion = max(0.0, min(1.0, dist_to_target / 120.0)) if dist_to_target > 0 else 1.0
    for ox, oy in obstaculos:
        dist = np.hypot(ox - rx, oy - ry)
        if 0 < dist < radio_colision:
            dx = rx - ox
            dy = ry - oy
            mag = np.hypot(dx, dy) + 1e-8
            penetracion = radio_colision - dist
            fuerza = (penetracion / radio_colision) * fuerza_max * factor_atenuacion
            fuerza = min(fuerza, 30)   # límite de fuerza
            tx += (dx / mag) * fuerza
            ty += (dy / mag) * fuerza
    return int(tx), int(ty)

def aplicar_repulsion_bordes(tx, ty, rx, ry, frame_w, frame_h, borde_seguridad, fuerza_borde):
    """
    Empuja el objetivo lejos de los bordes del campo.

    Parámetros:
        tx, ty (int): Coordenadas actuales del objetivo.
        rx, ry (int): Posición del robot.
        frame_w, frame_h (int): Dimensiones del campo en píxeles.
        borde_seguridad (int): Distancia mínima a los bordes.
        fuerza_borde (float): Magnitud de la fuerza de repulsión de bordes.

    Retorna:
        tuple: Nuevo objetivo (x,y) ajustado.
    """
    fuerza_total_x = 0
    fuerza_total_y = 0
    if rx < borde_seguridad:
        fuerza = (1.0 - rx / borde_seguridad) * fuerza_borde
        fuerza_total_x += fuerza
    if rx > frame_w - borde_seguridad:
        fuerza = (1.0 - (frame_w - rx) / borde_seguridad) * fuerza_borde
        fuerza_total_x -= fuerza
    if ry < borde_seguridad:
        fuerza = (1.0 - ry / borde_seguridad) * fuerza_borde
        fuerza_total_y += fuerza
    if ry > frame_h - borde_seguridad:
        fuerza = (1.0 - (frame_h - ry) / borde_seguridad) * fuerza_borde
        fuerza_total_y -= fuerza
    tx += fuerza_total_x
    ty += fuerza_total_y
    tx = max(0, min(frame_w, int(tx)))
    ty = max(0, min(frame_h, int(ty)))
    return tx, ty

class RobotPredictor:
    """
    Estima la posición y velocidad de un robot cuando se pierde de vista.
    """
    def __init__(self):
        self.last_pos = None
        self.velocity = (0, 0)
        self.lost_frames = 0
        self.filtered_pos = None

    def update(self, detected_pos):
        """
        Actualiza el predictor con una nueva posición detectada (o None).

        Parámetros:
            detected_pos (tuple o None): (x, y) si el robot fue visto, None en caso contrario.
        """
        if detected_pos is not None:
            x, y = detected_pos
            if self.last_pos is not None:
                vx = x - self.last_pos[0]
                vy = y - self.last_pos[1]
                alpha = 0.7
                self.velocity = (alpha * vx + (1-alpha)*self.velocity[0],
                                 alpha * vy + (1-alpha)*self.velocity[1])
            self.last_pos = (x, y)
            self.lost_frames = 0
            self.filtered_pos = (x, y)
        else:
            self.lost_frames += 1
            if self.last_pos is not None and self.lost_frames <= MAX_LOST_FRAMES:
                vx, vy = self.velocity
                px = self.last_pos[0] + vx * self.lost_frames
                py = self.last_pos[1] + vy * self.lost_frames
                self.filtered_pos = (int(px), int(py))
            else:
                self.filtered_pos = None

    def is_recovery(self):
        """Retorna True si el robot lleva mucho tiempo perdido."""
        return self.lost_frames > MAX_LOST_FRAMES_RECOVERY

# ============================================================
# --- MODIFICACIÓN: NUEVA FUNCIÓN DE CONVERSIÓN A PWM ---------
# ============================================================
def compute_pwm(turn, speed):
    """
    Convierte un par (turn, speed) en una cadena "pwm_izq,pwm_der".
    turn  : ángulo de giro (-90..90)
    speed : flag de velocidad (0 = parado, 1 = avanzar)
    Retorna:
        str con dos enteros separados por coma, ej. "150,150".
        Los valores pueden ser negativos para marcha atrás/giro en el sitio.
        Rango típico: -255 a 255.
    """
    if speed == 0:
        if turn != 0:
            # Giro en el sitio: una rueda hacia delante, otra hacia atrás
            base_turn = 100
            factor = 2.5
            left = int(np.clip(-turn * factor, -255, 255))
            right = int(np.clip(turn * factor, -255, 255))
            return f"{left},{right}"
        else:
            return "0,0"
    else:
        # Avance con diferencial
        base_speed = 150
        factor = 1.5
        left = int(np.clip(base_speed - turn * factor, -255, 255))
        right = int(np.clip(base_speed + turn * factor, -255, 255))
        return f"{left},{right}"

# ============================================================
# ESTRATEGIAS DE JUEGO
# ============================================================
# ----------------------------------------------------------
# ESTRATEGIA DEL PORTERO
# ----------------------------------------------------------
def strategy_portero(robot, ball, own_goal, opp_goal, w, h):
    """
    Estrategia del portero.
    
    Parámetros:
        robot  : dict {'pos': (x,y), 'angle': grados} del robot propio.
        ball   : tuple (x, y) de la posición de la pelota (puede ser None).
        own_goal: tuple (x, y) del centro de la portería propia.
        opp_goal: tuple (x, y) del centro de la portería rival.
        w, h   : int, ancho y alto del campo warp (750, 650).
    
    Retorna:
        comando (str)     : formato "<giro>,<velocidad>" (ej: "45,1"). 
                            Si no hay acción, retornar "0,0".
        punto_destino (tuple o None): coordenadas (x, y) para dibujar en la interfaz,
                                      o None si no se quiere pintar destino.
    """
    # ========== ESCRIBE AQUÍ LA ESTRATEGIA DEL PORTERO ==========
    # Ejemplo básico: regresar al arco si no hay bola
    if ball is None:
        # Aquí la lógica para reposicionarse en la portería
        pass

    # Si hay bola, defender la portería
    # ...

    # Por ahora retornamos sin movimiento y sin punto de destino
    return "0,0", None

# ----------------------------------------------------------
# ESTRATEGIA DEL MEDIOCAMPISTA
# ----------------------------------------------------------
def strategy_mediocampista(robot, ball, own_goal, opp_goal, w, h):
    """
    Estrategia del mediocampista.
    
    Parámetros:
        robot  : dict {'pos': (x,y), 'angle': grados} del robot propio.
        ball   : tuple (x, y) de la posición de la pelota (puede ser None).
        own_goal: tuple (x, y) del centro de la portería propia.
        opp_goal: tuple (x, y) del centro de la portería rival.
        w, h   : int, ancho y alto del campo warp (750, 650).
    
    Retorna:
        comando (str)     : formato "<giro>,<velocidad>" (ej: "45,1").
        punto_destino (tuple o None): coordenadas (x, y) para dibujar en la interfaz,
                                      o None si no se quiere pintar destino.
    """
    # ========== ESCRIBE AQUÍ LA ESTRATEGIA DEL MEDIOCAMPISTA ==========
    # Ejemplo básico: interponerse entre la pelota y la portería propia
    if ball is None:
        # Buscar posición defensiva
        pass

    # Si la pelota está en nuestro campo, presionar; si no, apoyar al delantero
    # ...

    # Por ahora retornamos sin movimiento y sin punto de destino
    return "0,0", None

# --- MODIFICACIÓN: strategy_delantero ahora retorna PWM ----------------
def strategy_delantero(robot, ball, own_goal, opp_goal, w, h,
                       obstaculos, predictor_del, frame_w, frame_h,
                       radio_colision, fuerza_max, borde_seguridad, fuerza_borde, distancia_detras_pelota):
    """
    Estrategia del delantero (funcional).
    - Calcula un punto detrás de la pelota en dirección al arco rival.
    - Aplica repulsión de obstáculos y bordes.
    - Usa un control PID para el giro.
    - Si el robot está en modo recuperación, gira lentamente.
    """
    if robot is None or ball is None:
        return "0,0", None

    # Modo recuperación (giro lento)
    if predictor_del.is_recovery():
        direccion = 1 if (time.time() * 2) % 2 < 1 else -1
        turn = direccion * 20
        return compute_pwm(turn, 0), None   # <-- PWM en lugar de "turn,0"

    bx, by = ball
    ox, oy = opp_goal
    dx = bx - ox
    dy = by - oy
    mag = np.hypot(dx, dy)
    if mag > 0:
        target_x = bx + (dx / mag) * distancia_detras_pelota
        target_y = by + (dy / mag) * distancia_detras_pelota
    else:
        target_x, target_y = bx, by

    rx, ry = robot['pos']
    dist_al_target = np.hypot(target_x - rx, target_y - ry)
    if dist_al_target < 20:
        target_x, target_y = bx, by

    # Repulsión
    target_x, target_y = aplicar_repulsion((rx, ry), (target_x, target_y), obstaculos,
                                           radio_colision, fuerza_max, dist_al_target)
    target_x, target_y = aplicar_repulsion_bordes(target_x, target_y, rx, ry,
                                                  frame_w, frame_h, borde_seguridad, fuerza_borde)

    # Control PID angular
    ang = robot['angle']
    ang_to_target = (np.degrees(np.arctan2(target_y - ry, target_x - rx)) + 360) % 360
    error_ang = ((ang_to_target - ang + 180) % 360) - 180

    # Variables estáticas para el PID (se inicializan la primera vez)
    if not hasattr(strategy_delantero, 'integral'):
        strategy_delantero.integral = 0.0
        strategy_delantero.prev_error = 0.0

    turn_pid, strategy_delantero.integral, strategy_delantero.prev_error = \
        control_pid_angular(error_ang, DT, strategy_delantero.integral, strategy_delantero.prev_error)

    turn = int(np.clip(turn_pid, -90, 90))
    velocidad = 1 if dist_al_target > 15 else 0

    return compute_pwm(turn, velocidad), (int(target_x), int(target_y))  # <-- PWM

# ============================================================
# LÓGICA PRINCIPAL DE JUEGO
# ============================================================
def calcular_logica(own_robots, ball_pos, opp_goal, own_goal, w, h,
                    obstaculos, predictors, frame_w, frame_h,
                    rep_config):
    """
    Calcula los comandos para los tres robots y devuelve la trama serial.

    Parámetros:
        own_robots (dict): Robots propios detectados {rol: {pos, angle}}.
        ball_pos (tuple): Posición de la pelota.
        opp_goal, own_goal (tuple): Posiciones de las porterías.
        w, h (int): Dimensiones del campo rectificado.
        obstaculos (list): Lista de posiciones de otros robots.
        predictors (dict): Predictores por robot.
        frame_w, frame_h (int): Ancho/alto de la imagen (mismo que w,h).
        rep_config (dict): Parámetros de repulsión.

    Retorna:
        str: Trama con comandos "<giro,vel|...>".
        dict: Puntos de depuración por robot.
    """
    comandos = {robot_POR: "0,0", robot_MED: "0,0", robot_DEL: "0,0"}
    debug_targets = {}

    for role in [robot_POR, robot_MED, robot_DEL]:
        robot = own_robots.get(role)
        if robot:
            if role == robot_POR:
                cmd, target = strategy_portero(robot, ball_pos, own_goal, opp_goal, w, h)
            elif role == robot_MED:
                cmd, target = strategy_mediocampista(robot, ball_pos, own_goal, opp_goal, w, h)
            elif role == robot_DEL:
                cmd, target = strategy_delantero(
                    robot, ball_pos, own_goal, opp_goal, w, h,
                    obstaculos, predictors[robot_DEL], frame_w, frame_h,
                    rep_config['RADIO_COLISION'],
                    rep_config['FUERZA_REPULSION_MAX'],
                    rep_config['BORDE_DIST_SEGURIDAD'],
                    rep_config['FUERZA_REPULSION_BORDE'],
                    rep_config['DISTANCIA_DETRAS_PELOTA']
                )
            comandos[role] = cmd
            if target is not None:
                debug_targets[role] = target

    trama = f"<{comandos[robot_POR]}|{comandos[robot_MED]}|{comandos[robot_DEL]}>\n"
    return trama, debug_targets

# ============================================================
# VENTANA DE CALIBRACIÓN (sin cambios)
# ============================================================
class CalibrationWindow(tk.Toplevel):
    def __init__(self, parent, camera_settings):
        super().__init__(parent)
        self.parent = parent
        self.camera_settings = camera_settings
        self.title("Calibración de Campo y Colores")
        self.geometry("800x650")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.camera = cv2.VideoCapture(camera_settings['index'], cv2.CAP_DSHOW)
        self._apply_camera_params()

        self.calibration_colors = ['naranja', 'amarillo', 'rosa', 'verde', 'azul']
        self.hsv_ranges = {}
        self.points = []               # Puntos de las esquinas del campo
        self.state = 'select_field'    # Máquina de estados de calibración
        self.raw_frame = None

        # Márgenes HSV que se irán ajustando
        self.H_MARGIN = DEFAULT_H_MARGIN
        self.S_MARGIN = DEFAULT_S_MARGIN
        self.V_MARGIN = DEFAULT_V_MARGIN

        self.current_image = None
        self.disp_offset_x = 0
        self.disp_offset_y = 0
        self.disp_w = 1
        self.disp_h = 1
        self.raw_img_w = 1
        self.raw_img_h = 1

        self.image_label = tk.Label(self)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Button-1>", self.mouse_callback)

        self.info_var = tk.StringVar()
        self._running = True
        self.update_info()
        self.update_frame()
        self.update()

    def _apply_camera_params(self):
        s = self.camera_settings
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, s['exposure'])
        self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)
        if s.get('auto_wb', False):
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, s.get('wb', 4000))
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, s['brightness'])
        self.camera.set(cv2.CAP_PROP_CONTRAST, s['contrast'])
        self.camera.set(cv2.CAP_PROP_SATURATION, s['saturation'])

    def update_info(self):
        if self.state == 'select_field':
            self.info_var.set("Haz clic en las 4 esquinas del campo (orden UL, UR, LR, LL)")
        elif self.state == 'calibrate':
            if len(self.hsv_ranges) < len(self.calibration_colors):
                next_color = self.calibration_colors[len(self.hsv_ranges)]
                self.info_var.set(f"Haz clic sobre el color: {next_color.upper()}")
            else:
                self.info_var.set("Todos los colores calibrados. Presiona 'Cerrar' para volver.")
        elif self.state == 'done':
            self.info_var.set("Calibración guardada. Puede cerrar esta ventana.")

    def mouse_callback(self, event):
        if self.state == 'select_field':
            if len(self.points) < 4:
                img_x = event.x - self.disp_offset_x
                img_y = event.y - self.disp_offset_y
                if img_x < 0 or img_x >= self.disp_w or img_y < 0 or img_y >= self.disp_h:
                    return
                scale_x = self.raw_img_w / self.disp_w
                scale_y = self.raw_img_h / self.disp_h
                x_real = int(img_x * scale_x)
                y_real = int(img_y * scale_y)
                self.points.append((x_real, y_real))
                if len(self.points) == 4:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    save_json(os.path.join(script_dir, 'cancha_config.json'), {'puntos_cancha': self.points})
                    self.state = 'calibrate'
                    self.update_info()

        elif self.state == 'calibrate' and self.raw_frame is not None:
            if len(self.hsv_ranges) < len(self.calibration_colors):
                img_x = event.x - self.disp_offset_x
                img_y = event.y - self.disp_offset_y
                if img_x < 0 or img_x >= self.disp_w or img_y < 0 or img_y >= self.disp_h:
                    return
                scale_x = self.raw_img_w / self.disp_w
                scale_y = self.raw_img_h / self.disp_h
                x_real = int(img_x * scale_x)
                y_real = int(img_y * scale_y)

                hsv_frame = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2HSV)
                # Pequeña ROI alrededor del clic para calcular el color medio
                y1, y2 = max(0, y_real - 2), min(hsv_frame.shape[0], y_real + 3)
                x1, x2 = max(0, x_real - 2), min(hsv_frame.shape[1], x_real + 3)
                roi_hsv = hsv_frame[y1:y2, x1:x2]
                mean_hsv = cv2.mean(roi_hsv)
                h, s, v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

                color_name = self.calibration_colors[len(self.hsv_ranges)]
                win_name = f"Ajuste fino {color_name}"
                cv2.namedWindow(win_name)
                cv2.createTrackbar('H marg', win_name, self.H_MARGIN, 30, lambda x: None)
                cv2.createTrackbar('S marg', win_name, self.S_MARGIN, 150, lambda x: None)
                cv2.createTrackbar('V marg', win_name, self.V_MARGIN, 150, lambda x: None)

                while True:
                    if not self._running:
                        break
                    h_m = cv2.getTrackbarPos('H marg', win_name)
                    s_m = cv2.getTrackbarPos('S marg', win_name)
                    v_m = cv2.getTrackbarPos('V marg', win_name)
                    lower = np.array([max(h - h_m, 0), max(s - s_m, 0), max(v - v_m, 0)])
                    upper = np.array([min(h + h_m, 180), min(s + s_m, 255), min(v + v_m, 255)])
                    mask = cv2.inRange(hsv_frame, lower, upper)
                    #cv2.imshow(win_name, mask)

                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    cv2.putText(mask_bgr, "Presiona 'g' para guardar",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(win_name, mask_bgr)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' ') or key == ord('g'):   # Teclas para guardar
                        self.H_MARGIN = h_m
                        self.S_MARGIN = s_m
                        self.V_MARGIN = v_m
                        break
                    elif key == 27:  # Escape para cancelar (no guarda márgenes)
                        break
                cv2.destroyWindow(win_name)

                lower = [max(h - self.H_MARGIN, 0),
                         max(s - self.S_MARGIN, 0),
                         max(v - self.V_MARGIN, 0)]
                upper = [min(h + self.H_MARGIN, 180),
                         min(s + self.S_MARGIN, 255),
                         min(v + self.V_MARGIN, 255)]
                self.hsv_ranges[color_name] = {'lower': lower, 'upper': upper}
                self.update_info()

                if len(self.hsv_ranges) == len(self.calibration_colors):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    save_json(os.path.join(script_dir, 'hsv_calibration.json'), self.hsv_ranges)
                    self.state = 'done'
                    self.update_info()

    def update_frame(self):
        if not self._running:
            return
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.raw_frame = frame.copy()
                display = frame.copy()

                if self.state == 'select_field':
                    for p in self.points:
                        cv2.circle(display, p, 4, (0, 255, 0), 1)
                        cv2.circle(display, p, 1, (0, 255, 255), -1)
                elif self.state in ['calibrate', 'done'] and len(self.points) == 4:
                    pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                if self.state != 'done':
                    msg = self.info_var.get()
                    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display, (10, display.shape[0]-th-25), (10+tw+10, display.shape[0]-10), (0,0,0), -1)
                    cv2.putText(display, msg, (15, display.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(rgb)
                self.show_image()
        self.after(30, self.update_frame)

    def show_image(self):
        if hasattr(self, 'current_image'):
            label_w = self.image_label.winfo_width()
            label_h = self.image_label.winfo_height()
            if label_w < 10 or label_h < 10:
                label_w, label_h = 640, 480
            img_w, img_h = self.current_image.size
            ratio = min(label_w / img_w, label_h / img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)

            self.disp_w = new_w
            self.disp_h = new_h
            self.disp_offset_x = (label_w - new_w) // 2
            self.disp_offset_y = (label_h - new_h) // 2
            if self.raw_frame is not None:
                self.raw_img_h, self.raw_img_w = self.raw_frame.shape[:2]

            resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img

    def on_close(self):
        self._running = False
        if self.camera.isOpened():
            self.camera.release()
        self.destroy()

# ============================================================
# APLICACIÓN PRINCIPAL (con barra de desplazamiento funcional)
# ============================================================
class App:
    BRIGHT_DEFAULT = 128
    CONTRAST_DEFAULT = 128
    SATURATION_DEFAULT = 128
    WB_DEFAULT = 4000

    def __init__(self, root):
        self.root = root
        self.root.title("RoboFútbol - Torneo Zacatecano VSSS")
        self.root.geometry("1200x800")

        self.camera_index = tk.IntVar(value=1)
        self.port_var = tk.StringVar(value="COM9")
        self.connected = False
        self.paused = False
        self.simulacion = True
        self.camera = None
        self.serial_conn = None

        self.own_goal_side = tk.StringVar(value="izquierda")
        self.my_color = tk.StringVar(value="azul")
        self.distancia_detras = tk.IntVar(value=25)
        self.show_raw = tk.BooleanVar(value=False)

        # Parámetros de repulsión ajustables
        self.radio_colision = tk.IntVar(value=35)
        self.fuerza_repulsion_max = tk.IntVar(value=80)
        self.borde_dist_seguridad = tk.IntVar(value=30)
        self.fuerza_repulsion_borde = tk.IntVar(value=90)

        self.auto_cam_var = tk.BooleanVar(value=False)

        # Corrección de distorsión radial (toggle)
        self.corregir_dist_var = tk.BooleanVar(value=False)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.field_config_path = os.path.join(script_dir, 'cancha_config.json')
        self.color_config_path = os.path.join(script_dir, 'hsv_calibration.json')
        self.calib_cam_path = os.path.join(script_dir, 'camera_calib.json')
        self.points = []
        self.colors_hsv = None

        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_camera_calibration()
        # Activar corrección por defecto si hay calibración de cámara
        if self.camera_matrix is not None:
            self.corregir_dist_var.set(True)

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.kalman = init_kalman()

        # Memoria de los robots: propios con roles, rivales como lista
        self.memoria_robots_own = {
            robot_POR: {'pos': None, 'angle': 0, 'lost_frames': 999},
            robot_MED: {'pos': None, 'angle': 0, 'lost_frames': 999},
            robot_DEL: {'pos': None, 'angle': 0, 'lost_frames': 999}
        }
        self.memoria_rivales = []   # Lista de dict con 'pos','angle'

        self.predictors = {
            robot_POR: RobotPredictor(),
            robot_MED: RobotPredictor(),
            robot_DEL: RobotPredictor()
        }
        self.frames_sin_bola = 999
        self.last_serial_time = time.time()
        self.last_trama = ""

        self.calibrating = False
        self.build_gui()
        self.load_field_config()
        self.load_color_config()
        self.open_camera()
        self.update()

    def load_camera_calibration(self):
        data = load_json(self.calib_cam_path)
        if data and 'camera_matrix' in data and 'dist_coeffs' in data:
            self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
            self.dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float32)
            print("Calibración de cámara cargada. Corrección de distorsión activada.")
        else:
            self.camera_matrix = None
            self.dist_coeffs = None
            print("Sin calibración de cámara. No se corregirá la distorsión radial.")

    # --- MÉTODOS AUXILIARES PARA LA BARRA DE DESPLAZAMIENTO (NUEVOS) ---
    def _on_control_frame_configure(self, event):
        """Actualiza el scrollregion cuando cambia el tamaño del frame interior."""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Ajusta el ancho del control_frame al ancho del canvas."""
        canvas_width = event.width
        self.left_canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _on_mousewheel(self, event):
        """Desplaza el canvas con la rueda del ratón."""
        self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # --- CONSTRUCCIÓN DE LA GUI (MODIFICADA PARA SCROLL) ----------------
    def build_gui(self):
        # Contenedor izquierdo con canvas y scrollbar
        left_container = tk.Frame(self.root, width=300)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        left_container.pack_propagate(False)

        # Canvas y scrollbar
        self.left_canvas = tk.Canvas(left_container, width=300, highlightthickness=0,
                                     bg=self.root.cget('bg'))   # fondo igual a la ventana
        self.left_scrollbar = tk.Scrollbar(left_container, orient=tk.VERTICAL,
                                           command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame interior que contiene todos los controles
        self.control_frame = tk.Frame(self.left_canvas, bg=self.root.cget('bg'))
        self.canvas_window = self.left_canvas.create_window((0, 0), window=self.control_frame,
                                                            anchor=tk.NW)

        # Eventos para el scroll
        self.control_frame.bind("<Configure>", self._on_control_frame_configure)
        self.left_canvas.bind("<Configure>", self._on_canvas_configure)
        # La rueda del ratón solo funciona dentro del canvas
        self.left_canvas.bind("<Enter>", lambda e: self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.left_canvas.bind("<Leave>", lambda e: self.left_canvas.unbind_all("<MouseWheel>"))

        # --- Widgets del panel (idénticos a tu código original) ---
        tk.Label(self.control_frame, text="Cámara:").grid(row=0, column=0, sticky='w')
        self.cam_combo = ttk.Combobox(self.control_frame, textvariable=self.camera_index,
                                      values=[0,1,2], state='readonly', width=5)
        self.cam_combo.grid(row=0, column=1, sticky='w')
        self.cam_combo.bind('<<ComboboxSelected>>', self.on_camera_change)

        tk.Label(self.control_frame, text="Puerto COM:").grid(row=1, column=0, sticky='w', pady=(10,0))
        self.port_combo = ttk.Combobox(self.control_frame, textvariable=self.port_var,
                                       values=["COM3","COM4","COM5","COM6","COM7","COM8","COM9","COM10",
                                               "/dev/ttyUSB0","/dev/ttyUSB1"], width=13)
        self.port_combo.grid(row=1, column=1, sticky='w', pady=(10,0))
        self.conn_btn = tk.Button(self.control_frame, text="Conectar", command=self.toggle_connection,
                                  bg="lightgray")
        self.conn_btn.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)

        self.pause_btn = tk.Button(self.control_frame, text="Pausa", command=self.toggle_pause,
                                   bg="lightgray")
        self.pause_btn.grid(row=3, column=0, columnspan=2, sticky='ew', pady=5)

        self.calib_btn = tk.Button(self.control_frame, text="Calibración", command=self.open_calibration)
        self.calib_btn.grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)

        self.raw_check = tk.Checkbutton(self.control_frame, text="Vista Base (RAW)",
                                        variable=self.show_raw)
        self.raw_check.grid(row=5, column=0, columnspan=2, sticky='w', pady=(10,0))

        tk.Label(self.control_frame, text="Portería propia:").grid(row=6, column=0, sticky='w',
                                                                   pady=(10,0))
        self.goal_combo = ttk.Combobox(self.control_frame, textvariable=self.own_goal_side,
                                       values=["izquierda", "derecha"], state='readonly', width=10)
        self.goal_combo.grid(row=6, column=1, sticky='w', pady=(10,0))

        tk.Label(self.control_frame, text="Color uniforme:").grid(row=7, column=0, sticky='w', pady=(5,0))
        self.color_combo = ttk.Combobox(self.control_frame, textvariable=self.my_color,
                                        values=["azul", "amarillo"], state='readonly', width=10)
        self.color_combo.grid(row=7, column=1, sticky='w', pady=(5,0))

        tk.Label(self.control_frame, text="Distancia detrás pelota:").grid(row=8, column=0, sticky='w',
                                                                            pady=(10,0))
        self.dist_slider = tk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                    variable=self.distancia_detras, length=150)
        self.dist_slider.grid(row=9, column=0, columnspan=2, sticky='w')

        # Repulsión robots
        tk.Label(self.control_frame, text="Repulsión robots",
                 font=('Arial', 9, 'bold')).grid(row=10, column=0, columnspan=2, sticky='w',
                                                 pady=(10,0))
        tk.Label(self.control_frame, text="Radio (px):").grid(row=11, column=0, sticky='w')
        self.radio_slider = tk.Scale(self.control_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                                     variable=self.radio_colision, length=150)
        self.radio_slider.grid(row=11, column=1, sticky='w')
        tk.Label(self.control_frame, text="Fuerza:").grid(row=12, column=0, sticky='w')
        self.fuerza_robot_slider = tk.Scale(self.control_frame, from_=10, to=150, orient=tk.HORIZONTAL,
                                            variable=self.fuerza_repulsion_max, length=150)
        self.fuerza_robot_slider.grid(row=12, column=1, sticky='w')

        # Repulsión bordes
        tk.Label(self.control_frame, text="Repulsión bordes",
                 font=('Arial', 9, 'bold')).grid(row=13, column=0, columnspan=2, sticky='w',
                                                 pady=(10,0))
        tk.Label(self.control_frame, text="Margen (px):").grid(row=14, column=0, sticky='w')
        self.borde_slider = tk.Scale(self.control_frame, from_=10, to=80, orient=tk.HORIZONTAL,
                                     variable=self.borde_dist_seguridad, length=150)
        self.borde_slider.grid(row=14, column=1, sticky='w')
        tk.Label(self.control_frame, text="Fuerza:").grid(row=15, column=0, sticky='w')
        self.fuerza_borde_slider = tk.Scale(self.control_frame, from_=10, to=150, orient=tk.HORIZONTAL,
                                            variable=self.fuerza_repulsion_borde, length=150)
        self.fuerza_borde_slider.grid(row=15, column=1, sticky='w')

        # Parámetros de cámara
        tk.Label(self.control_frame, text="Parámetros de cámara",
                 font=('Arial', 10, 'bold')).grid(row=16, column=0, columnspan=2, sticky='w',
                                                  pady=(15,5))
        self.exp_var = tk.DoubleVar(value=-6)
        tk.Label(self.control_frame, text="Exposición").grid(row=17, column=0, sticky='w')
        self.exp_slider = tk.Scale(self.control_frame, from_=-13, to=-1, resolution=0.1,
                                   orient=tk.HORIZONTAL, variable=self.exp_var, length=150,
                                   command=self.on_param_change)
        self.exp_slider.grid(row=17, column=1, sticky='w')

        self.bright_var = tk.DoubleVar(value=128)
        tk.Label(self.control_frame, text="Brillo").grid(row=18, column=0, sticky='w')
        self.bright_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                      variable=self.bright_var, length=150,
                                      command=self.on_param_change)
        self.bright_slider.grid(row=18, column=1, sticky='w')

        self.contrast_var = tk.DoubleVar(value=128)
        tk.Label(self.control_frame, text="Contraste").grid(row=19, column=0, sticky='w')
        self.contrast_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        variable=self.contrast_var, length=150,
                                        command=self.on_param_change)
        self.contrast_slider.grid(row=19, column=1, sticky='w')

        self.saturation_var = tk.DoubleVar(value=128)
        tk.Label(self.control_frame, text="Saturación").grid(row=20, column=0, sticky='w')
        self.saturation_slider = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                          variable=self.saturation_var, length=150,
                                          command=self.on_param_change)
        self.saturation_slider.grid(row=20, column=1, sticky='w')

        self.wb_var = tk.DoubleVar(value=4000)
        tk.Label(self.control_frame, text="White Balance").grid(row=21, column=0, sticky='w')
        self.wb_slider = tk.Scale(self.control_frame, from_=2800, to=6500, orient=tk.HORIZONTAL,
                                  variable=self.wb_var, length=150, command=self.on_param_change)
        self.wb_slider.grid(row=21, column=1, sticky='w')

        # Checkbox para ajustes automáticos de cámara
        self.auto_cam_chk = tk.Checkbutton(self.control_frame, text="Auto (Exp manual)",
                                           variable=self.auto_cam_var,
                                           command=self.toggle_auto_camera)
        self.auto_cam_chk.grid(row=22, column=0, columnspan=2, sticky='w', pady=(10,0))

        # Botón de corrección de distorsión radial (debajo del auto_cam)
        self.dist_btn = tk.Button(self.control_frame, text="Corregir distorsión radial",
                                  command=self.toggle_corregir_distorsion,
                                  relief=tk.RAISED)
        self.dist_btn.grid(row=23, column=0, columnspan=2, sticky='ew', pady=(8,0))
        self._update_dist_button()

        # Área de visualización de la cámara (derecha)
        self.display_frame = tk.Frame(self.root, bg='black')
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_frame.bind("<Configure>", self.on_resize)

        self.image_label = tk.Label(self.display_frame, bg='black')
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def toggle_corregir_distorsion(self):
        """Activa/desactiva la corrección de distorsión y actualiza el botón."""
        self.corregir_dist_var.set(not self.corregir_dist_var.get())
        self._update_dist_button()

    def _update_dist_button(self):
        if self.corregir_dist_var.get():
            self.dist_btn.config(relief=tk.SUNKEN, text="Corregir distorsión radial (ON)")
        else:
            self.dist_btn.config(relief=tk.RAISED, text="Corregir distorsión radial (OFF)")

    def load_field_config(self):
        data = load_json(self.field_config_path)
        if data and 'puntos_cancha' in data:
            self.points = data['puntos_cancha']
        else:
            self.points = []

    def load_color_config(self):
        data = load_json(self.color_config_path)
        if data:
            self.colors_hsv = {}
            for color, ranges in data.items():
                self.colors_hsv[color] = (np.array(ranges['lower']), np.array(ranges['upper']))
        else:
            self.colors_hsv = {
                'naranja': (np.array([0, 120, 70]), np.array([20, 255, 255])),
                'amarillo': (np.array([20, 100, 100]), np.array([40, 255, 255])),
                'rosa': (np.array([140, 50, 50]), np.array([170, 255, 255])),
                'verde': (np.array([40, 50, 50]), np.array([80, 255, 255])),
                'azul': (np.array([100, 50, 50]), np.array([130, 255, 255]))
            }

    def open_camera(self):
        if self.camera is not None:
            self.camera.release()
        idx = self.camera_index.get()
        self.camera = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.apply_camera_params()
        if not self.camera.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la cámara {idx}")

    def apply_camera_params(self, event=None):
        if self.camera is None or not self.camera.isOpened():
            return
        try:
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exp_var.get())
        except:
            pass
        if not self.auto_cam_var.get():
            props = {
                cv2.CAP_PROP_BRIGHTNESS: self.bright_var.get(),
                cv2.CAP_PROP_CONTRAST: self.contrast_var.get(),
                cv2.CAP_PROP_SATURATION: self.saturation_var.get(),
                cv2.CAP_PROP_AUTO_WB: 0,
                cv2.CAP_PROP_WB_TEMPERATURE: self.wb_var.get()
            }
            for prop, val in props.items():
                try:
                    self.camera.set(prop, val)
                except:
                    pass
        else:
            try:
                self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
            except:
                pass

    def toggle_auto_camera(self):
        auto = self.auto_cam_var.get()
        if auto:
            self.bright_var.set(self.BRIGHT_DEFAULT)
            self.contrast_var.set(self.CONTRAST_DEFAULT)
            self.saturation_var.set(self.SATURATION_DEFAULT)
            self.wb_var.set(self.WB_DEFAULT)
            state = tk.DISABLED
        else:
            state = tk.NORMAL
        self.bright_slider.config(state=state)
        self.contrast_slider.config(state=state)
        self.saturation_slider.config(state=state)
        self.wb_slider.config(state=state)
        self.apply_camera_params()

    def on_camera_change(self, event=None):
        self.open_camera()

    def on_param_change(self, event=None):
        self.apply_camera_params()

    def toggle_connection(self):
        if self.connected:
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = None
            self.connected = False
            self.simulacion = True
            self.conn_btn.config(text="Conectar", bg="lightgray")
        else:
            port = self.port_var.get()
            try:
                self.serial_conn = serial.Serial(port, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
                self.serial_conn.setDTR(False)
                self.serial_conn.setRTS(False)
                time.sleep(2)
                self.connected = True
                self.simulacion = False
                self.conn_btn.config(text="Desconectar", bg="lightgreen")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo conectar a {port}:\n{e}")

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Reanudar", bg="orange")
            if self.connected and self.serial_conn:
                try:
                    self.serial_conn.write("<0,0|0,0|0,0>\n".encode())
                except:
                    pass
        else:
            self.pause_btn.config(text="Pausa", bg="lightgray")

    def open_calibration(self):
        if self.camera is None or not self.camera.isOpened():
            messagebox.showerror("Error", "La cámara no está activa.")
            return
        camera_settings = {
            'index': self.camera_index.get(),
            'exposure': self.exp_var.get(),
            'brightness': self.bright_var.get(),
            'contrast': self.contrast_var.get(),
            'saturation': self.saturation_var.get(),
            'wb': self.wb_var.get(),
            'auto_wb': self.auto_cam_var.get()
        }
        self.calibrating = True
        self.camera.release()
        cal_win = CalibrationWindow(self.root, camera_settings)
        self.root.wait_window(cal_win)
        self.calibrating = False
        self.open_camera()
        self.reload_configs()

    def reload_configs(self):
        self.load_field_config()
        self.load_color_config()
        self.load_camera_calibration()
        # Actualizar botón de corrección por si la calibración cambió
        if self.camera_matrix is not None:
            self.corregir_dist_var.set(True)
        else:
            self.corregir_dist_var.set(False)
        self._update_dist_button()

    def update(self):
        if self.calibrating:
            self.root.after(30, self.update)
            return

        if self.camera is None or not self.camera.isOpened():
            self.root.after(30, self.update)
            return

        ret, frame = self.camera.read()
        if not ret:
            self.root.after(30, self.update)
            return

        # Aplicar corrección de distorsión radial según el botón
        if self.corregir_dist_var.get():
            frame = undistort_frame(frame, self.camera_matrix, self.dist_coeffs)

        if len(self.points) == 4 and self.colors_hsv is not None:
            warped, w, h = warp_image(frame, self.points)
            if self.show_raw.get():
                display_img = frame.copy()
                self.draw_hud(display_img, None, None, frame.shape[1], frame.shape[0])
            else:
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                hh, s, v = cv2.split(hsv)
                v = self.clahe.apply(v)
                hsv_enhanced = cv2.merge((hh, s, v))
                warped_display = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
                display_img = warped_display.copy()

                ball_pos = detect_ball(hsv_enhanced, self.colors_hsv['naranja'])
                pred = self.kalman.predict()
                if ball_pos:
                    self.kalman.correct(np.array([[ball_pos[0]], [ball_pos[1]]], np.float32))
                    self.frames_sin_bola = 0
                    target_ball = ball_pos
                else:
                    self.frames_sin_bola += 1
                    if self.frames_sin_bola < 30:
                        target_ball = (int(pred[0][0]), int(pred[1][0]))
                    else:
                        target_ball = None

                # Identificación de robots
                own_robots, rivales = identify_team(hsv_enhanced, self.colors_hsv, self.my_color.get())

                # Actualizar memoria de robots propios y predictores
                for role in [robot_POR, robot_MED, robot_DEL]:
                    detected = own_robots.get(role)
                    if detected is not None:
                        pos = detected['pos']
                        ang = detected['angle']
                        self.predictors[role].update(pos)
                        self.memoria_robots_own[role]['pos'] = pos
                        self.memoria_robots_own[role]['angle'] = ang
                        self.memoria_robots_own[role]['lost_frames'] = 0
                    else:
                        self.predictors[role].update(None)
                        filtered = self.predictors[role].filtered_pos
                        self.memoria_robots_own[role]['pos'] = filtered
                        self.memoria_robots_own[role]['lost_frames'] = self.predictors[role].lost_frames

                # Almacenar rivales como lista
                self.memoria_rivales = rivales

                self.draw_field_references(display_img, w, h)

                # Robots propios activos (los que se están viendo o se recuerdan)
                own_active = {}
                lista_obstaculos = []
                for role in [robot_POR, robot_MED, robot_DEL]:
                    mem = self.memoria_robots_own[role]
                    if mem['lost_frames'] < MEMORY_FRAMES and mem['pos'] is not None:
                        own_active[role] = {'pos': mem['pos'], 'angle': mem['angle']}
                        self.draw_own_robot(display_img, role, mem['pos'], mem['angle'])
                        lista_obstaculos.append(mem['pos'])

                # Añadir rivales como obstáculos y dibujarlos
                for rival in rivales:
                    if rival['pos'] is not None:
                        lista_obstaculos.append(rival['pos'])
                        self.draw_rival_robot(display_img, rival['pos'])

                # Dibujar zonas de repulsión
                self.draw_repulsion_zones(display_img, w, h, lista_obstaculos)

                if self.own_goal_side.get() == 'izquierda':
                    own_goal = (0, h // 2)
                    opp_goal = (w, h // 2)
                else:
                    own_goal = (w, h // 2)
                    opp_goal = (0, h // 2)

                if target_ball is not None:
                    rep_config = {
                        'RADIO_COLISION': self.radio_colision.get(),
                        'FUERZA_REPULSION_MAX': self.fuerza_repulsion_max.get(),
                        'BORDE_DIST_SEGURIDAD': self.borde_dist_seguridad.get(),
                        'FUERZA_REPULSION_BORDE': self.fuerza_repulsion_borde.get(),
                        'DISTANCIA_DETRAS_PELOTA': self.distancia_detras.get()
                    }
                    trama, debug_targets = calcular_logica(
                        own_active, target_ball, opp_goal, own_goal,
                        w, h, lista_obstaculos, self.predictors, w, h, rep_config
                    )
                    for robot, target in debug_targets.items():
                        cv2.circle(display_img, target, 8, COLOR_TARGET, 2)
                    self.send_serial_if_needed(trama)
                    self.draw_hud(display_img, trama, target_ball, w, h)
                else:
                    self.draw_hud(display_img, "<ESPERANDO BOLA>", None, w, h)

                if target_ball:
                    cv2.circle(display_img, target_ball, 8, COLOR_BALL, -1)

            self.display_image(display_img)
        else:
            display_img = frame.copy()
            cv2.putText(display_img, "Calibracion necesaria: haz clic en Calibracion", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.display_image(display_img)

        self.root.after(30, self.update)

    def send_serial_if_needed(self, trama):
        if self.simulacion or self.paused or not self.connected:
            return
        current_time = time.time()
        if (trama != self.last_trama and current_time - self.last_serial_time > 0.05) or \
           (current_time - self.last_serial_time > 0.5):
            try:
                self.serial_conn.write(trama.encode())
                self.last_serial_time = current_time
                self.last_trama = trama
            except:
                self.connected = False
                self.simulacion = True
                self.serial_conn = None
                self.conn_btn.config(text="Conectar", bg="lightgray")

    def draw_repulsion_zones(self, img, w, h, obstaculos):
        radio = self.radio_colision.get()
        borde = self.borde_dist_seguridad.get()
        for (rx, ry) in obstaculos:
            cv2.circle(img, (rx, ry), radio, COLOR_REPULSION, 1)
        cv2.line(img, (borde, 0), (borde, h), COLOR_REPULSION, 1, cv2.LINE_AA)
        cv2.line(img, (w - borde, 0), (w - borde, h), COLOR_REPULSION, 1, cv2.LINE_AA)
        cv2.line(img, (0, borde), (w, borde), COLOR_REPULSION, 1, cv2.LINE_AA)
        cv2.line(img, (0, h - borde), (w, h - borde), COLOR_REPULSION, 1, cv2.LINE_AA)

    def draw_field_references(self, img, w, h):
        if self.own_goal_side.get() == 'izquierda':
            own = (0, h//2)
            opp = (w, h//2)
        else:
            own = (w, h//2)
            opp = (0, h//2)
        cv2.circle(img, own, 20, (0,255,255), 2)
        cv2.line(img, (own[0]-20, own[1]), (own[0]+20, own[1]), (0,255,255), 2)
        cv2.line(img, (own[0], own[1]-20), (own[0], own[1]+20), (0,255,255), 2)
        cv2.circle(img, opp, 20, (0,0,255), 2)
        cv2.line(img, (opp[0]-20, opp[1]), (opp[0]+20, opp[1]), (0,0,255), 2)
        cv2.line(img, (opp[0], opp[1]-20), (opp[0], opp[1]+20), (0,0,255), 2)

        area_width_px = 350
        area_depth_px = 75
        goal_y = h // 2
        top_left_y = goal_y - area_width_px // 2
        bottom_right_y = goal_y + area_width_px // 2
        if self.own_goal_side.get() == 'izquierda':
            cv2.rectangle(img, (0, top_left_y), (area_depth_px, bottom_right_y), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (w - area_depth_px, top_left_y), (w, bottom_right_y), (255, 0, 0), 2)

    def draw_own_robot(self, img, role, pos, angle):
        x, y = pos
        if role == robot_POR:
            color = COLOR_POR_OWN
        elif role == robot_MED:
            color = COLOR_MED_OWN
        elif role == robot_DEL:
            color = COLOR_DEL_OWN
        else:
            color = (0, 255, 0)
        cv2.circle(img, (x,y), 15, color, 2)
        cv2.putText(img, role, (x-15, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        rad = np.radians(angle)
        dx = int(15 * np.cos(rad))
        dy = int(15 * np.sin(rad))
        cv2.arrowedLine(img, (x,y), (x+dx, y+dy), color, 2, tipLength=0.3)

    def draw_rival_robot(self, img, pos):
        x, y = pos
        cv2.circle(img, (x,y), 15, COLOR_RIVAL, 2)
        # Solo se muestra la etiqueta "Rival", sin rol
        cv2.putText(img, "Rival", (x-30, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RIVAL, 2)

    def draw_hud(self, img, trama, ball_pos, frame_w, frame_h):
        overlay = img.copy()
        panel_w, panel_h = 350, 130
        x0, y0 = 5, frame_h - panel_h - 5
        cv2.rectangle(overlay, (x0, y0), (x0+panel_w, frame_h-5), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        y = frame_h - 20
        if self.simulacion:
            txt = "MODO SIMULACION"
        elif self.paused:
            txt = "PAUSA - Comandos detenidos"
        else:
            txt = f"Enviando: {trama.strip() if trama else 'N/A'}"
        cv2.putText(img, txt, (15, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        if ball_pos:
            cv2.putText(img, f"Pelota: X:{ball_pos[0]} Y:{ball_pos[1]}", (15, y-55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            cv2.putText(img, "Pelota no encontrada", (15, y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        estado = "PAUSA ACTIVA" if self.paused else "ACTIVO"
        cv2.putText(img, f"Estado: {estado}", (15, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        conn_txt = "Conectado" if self.connected else "Simulacion"
        cv2.putText(img, f"Conexion: {conn_txt}", (15, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image_pil = Image.fromarray(rgb)
        self.show_image()

    def show_image(self):
        if not hasattr(self, 'current_image_pil'):
            return
        label_w = self.image_label.winfo_width()
        label_h = self.image_label.winfo_height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 640, 480
        img_w, img_h = self.current_image_pil.size
        ratio = min(label_w/img_w, label_h/img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        resized = self.current_image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

    def on_resize(self, event):
        self.show_image()

    def on_close(self):
        if self.serial_conn:
            try:
                self.serial_conn.write("<0,0|0,0|0,0>\n".encode())
                time.sleep(0.1)
                self.serial_conn.close()
            except:
                pass
        if self.camera and self.camera.isOpened():
            self.camera.release()
        self.root.destroy()

# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()