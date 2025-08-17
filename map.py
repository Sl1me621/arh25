#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# Размеры поля (метры)
FIELD_WIDTH_M  = 7.975   # ось x карты: влево от правого верхнего угла
FIELD_HEIGHT_M = 5.875   # ось y карты: вниз от правого верхнего угла

# Палитра (BGR)
COLOR_MAP = {
    'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0),
    'cyan': (255,255,0), 'magenta': (255,0,255), 'yellow': (0,255,255),
    'white': (255,255,255), 'black': (0,0,0)
}

# --------- Нормализация осей дрона ---------
# Тогда XR = -y_d, YD = x_d.
# --------- Нормализация осей дрона ---------
# Дрон: y -> вправо, x -> вниз
# Нормализуем в систему "XR вправо, YD вниз"
def drone_to_right_down(x_d, y_d):
    xr = y_d
    yd = x_d
    return (xr, yd)

# --------- Гомография: дрон -> метры карты ---------
def compute_homography_drone_to_map(drone_corners):
    """
    drone_corners: словарь углов карты в СК дрона:
      {
        'tl': (x_d, y_d),  # слева-сверху
        'bl': (x_d, y_d),  # слева-снизу
        'br': (x_d, y_d),  # справа-снизу
        'tr': (x_d, y_d),  # справа-сверху
      }
    Возвращает матрицу H: [XR,YD,1] -> [x_map_m, y_map_m, 1],
    где (x_map_m, y_map_m) — метры карты с (0,0) в правом верхнем, x влево, y вниз.
    """
    # Нормализуем углы из СК дрона в XR-вправо/YD-вниз
    tl_rd = drone_to_right_down(*drone_corners['tl'])
    bl_rd = drone_to_right_down(*drone_corners['bl'])
    br_rd = drone_to_right_down(*drone_corners['br'])
    tr_rd = drone_to_right_down(*drone_corners['tr'])

    # Источник: порядок (tr, tl, bl, br) — чтобы удобно положить приёмник с (0,0) в ПРАВОМ ВЕРХНЕМ
    src = np.array([tr_rd, tl_rd, bl_rd, br_rd], dtype=np.float32)

    # Приёмник (метры карты):
    # tr -> (0,0), tl -> (W,0), bl -> (W,H), br -> (0,H)
    dst = np.array([
        [0.0,           0.0],            # top-right
        [FIELD_WIDTH_M, 0.0],            # top-left (x растёт влево)
        [FIELD_WIDTH_M, FIELD_HEIGHT_M], # bottom-left
        [0.0,           FIELD_HEIGHT_M], # bottom-right
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    return H

def transform_points(H, pts_rd):
    """Применить гомографию к списку точек в системе XR-вправо/YD-вниз."""
    pts_np = np.array(pts_rd, dtype=np.float32).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(pts_np, H)
    return warped.reshape(-1,2)

# --------- Метры карты -> пиксели изображения ---------
# На карте (0,0) — ПРАВЫЙ ВЕРХНИЙ угол, x растёт ВЛЕВО, y — ВНИЗ.
def map_meters_to_pixels(x_m, y_m, img_w, img_h):
    px = int(round(img_w - (x_m / FIELD_WIDTH_M) * img_w))  # от правого края влево
    py = int(round((y_m / FIELD_HEIGHT_M) * img_h))         # сверху вниз
    return px, py

def draw_label(img, text, org, font_scale=0.5, text_color=(255,255,255),
               bg_color=(30,30,30), thickness=1, pad=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    x,y = org
    cv2.rectangle(img, (x-pad, y-th-base-pad), (x+tw+pad, y+pad), bg_color, -1)
    cv2.putText(img, text, (x, y-base), font, font_scale, text_color, thickness, cv2.LINE_AA)

def annotate_map_from_drone(map_path, drone_corners, drone_objects, out_path="map_from_drone.png"):
    """
    drone_corners: {'tl':(x_d,y_d), 'bl':(x_d,y_d), 'br':(x_d,y_d), 'tr':(x_d,y_d)}  — в СК дрона
    drone_objects: [{'x':x_d, 'y':y_d, 'color':'red'|...}, ...] — в СК дрона
    """
    img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    assert img is not None, f"Не удалось открыть изображение карты: {map_path}"
    h, w = img.shape[:2]

    # 1) Гомография дрон -> метры карты
    H = compute_homography_drone_to_map(drone_corners)

    # 2) Нормализация осей точек дрона и трансформация
    pts_rd = [drone_to_right_down(obj['x'], obj['y']) for obj in drone_objects]
    map_pts = transform_points(H, pts_rd)

    # 3) Рисуем точки и подписи
    for (x_m, y_m), obj in zip(map_pts, drone_objects):
        color_name = obj.get('color','red').lower().strip()
        bgr = COLOR_MAP.get(color_name, (0,0,255))

        px, py = map_meters_to_pixels(x_m, y_m, w, h)

        # круг с контуром для читаемости
        cv2.circle(img, (px,py), 20, (0,0,0), -1)
        cv2.circle(img, (px,py), 20, bgr, -1)
        cv2.circle(img, (px,py), 20, (255,255,255), 1)
        x_label = FIELD_WIDTH_M - x_m  # зеркалим X относительно правого края
        y_label = y_m                   # Y совпадает (оба вниз)
        label = f"({x_label:.2f}; {y_label:.2f}) {color_name}"
        lx = min(px + 12, w-5)
        ly = max(py - 12, 18)
        draw_label(img, label, (lx, ly))

    cv2.imwrite(out_path, img)
    print("[OK] Сохранено:", os.path.abspath(out_path))
    return out_path

# ------------------ пример использования ------------------
if __name__ == "__main__":
    MAP_PATH = "map.jpg"  # твой файл карты

    # Углы карты в СК дрона (как ты описал словами):
    drone_corners = {
        'tl': (-0.8, -5.0),  # слева-сверху
        'bl': ( 4.5, -5.0),  # слева-снизу
        'br': ( 4.5,  2.0),  # справа-снизу
        'tr': (-0.8,  2.2),  # справа-сверху
    }

    # Пример точек (в СК дрона)
    drone_objects = [
        {'x': -0.05, 'y': -4.06, 'color': 'red'},
        {'x':  1.90, 'y': -4.18, 'color': 'green'},
        {'x':  3.20, 'y': -4.20, 'color': 'red'},
        {'x':  3.70, 'y': -2.15, 'color': 'blue'},
        {'x':  2.10, 'y': -2.00, 'color': 'green'},
        {'x':  0.05, 'y': -1.90, 'color': 'blue'},
        {'x':  0.17, 'y': -0.85, 'color': 'green'},
        {'x':  0.75, 'y': -0.35, 'color': 'red'},
        {'x':  3.65, 'y':  1.15, 'color': 'red'},
        {'x':  1.80, 'y':  1.35, 'color': 'green'},
        {'x':  0.17, 'y':  1.40, 'color': 'blue'}
    ]

    annotate_map_from_drone(MAP_PATH, drone_corners, drone_objects, out_path="map_from_drone.png")
