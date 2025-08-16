#!/usr/bin/env python3
import cv2
import socket
import struct
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

class DroneColorDetector:
    def __init__(self, server_ip='192.168.0.6', port=5001):
        self.server_ip = server_ip
        self.port = port
        self.client_socket = None
        
        # Параметры цветового детектирования (HSV)
        self.detection_history = []
        self.reappear_gap = 200
        
        self.color_params = {
            'red': {
                'lower': np.array([133, 154, 0]),
                'upper': np.array([179, 255, 255]),
                'color': (0, 0, 255)
            },
            'green': {
                   'lower': np.array([33, 104, 15]),
                    'upper': np.array([77, 209, 255]),
                    'color': (0, 0, 255)  # Красный в BGR
            },
            'blue': {
                # 'lower': np.array([75, 82, 142]),
                # 'upper': np.array([117, 255, 255]),
                # 'color': (0, 0, 255)  # Красный в BGR
                # 'lower': np.array([65, 42, 160]),
                # 'upper': np.array([113, 155, 255]),
                # 'color': (0, 0, 255)  # Красный в BGR

                'lower': np.array([64, 17, 100]),
                'upper': np.array([100, 147, 255]),
                'color': (0, 0, 255)  # Красный в BGR
            }
        }
        
        # Параметры фильтрации
        self.min_area = 200
        self.max_area = 1000
        self.circularity_thresh = 0.7
        self.min_detections = 20
        self.max_distance = 150
        
        # Трекинг объектов
        self.object_id = 0
        self.tracked_objects = {}
        self.detection_stats = defaultdict(int)
        
        # Статистика
        self.frame_count = 0
        self.start_time = time.time()
        self.recording = False
        self.video_writer = None

    def connect(self):
        """Установка соединения с сервером"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.port))
        print(f"Connected to {self.server_ip}:{self.port}")

    def init_video_writer(self, frame_size):
        """Инициализация записи видео"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drone_processed_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.recording = True
        print(f"Recording started: {filename}")

    def is_circular(self, cnt):
        """Проверка округлости контура"""
        area = cv2.contourArea(cnt)
        if area == 0:
            return False
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > self.circularity_thresh

    def _get_nearest_object(self, point, color):
        """Поиск ближайшего объекта для трекинга"""
        min_dist = float('inf')
        nearest_id = None
        
        for obj_id, obj_info in self.tracked_objects.items():
            if obj_info['color'] != color:
                continue
            if self.frame_count - obj_info['last_seen'] > self.reappear_gap:
                continue
                
            last_pos = obj_info['positions'][-1]
            dist = np.linalg.norm(np.array(point) - np.array(last_pos))
            
            if dist < self.max_distance and dist < min_dist:
                min_dist = dist
                nearest_id = obj_id
                
        return nearest_id if min_dist < self.max_distance else None

    def _update_object_tracking(self, current_detections):
        """Обновление информации о трекинге объектов"""
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['detected'] = False
        
        for detection in current_detections:
            center = detection['center']
            color = detection['color']
            
            obj_id = self._get_nearest_object(center, color)
            
            if obj_id is not None:
                self.tracked_objects[obj_id]['positions'].append(center)
                self.tracked_objects[obj_id]['last_seen'] = self.frame_count
                self.tracked_objects[obj_id]['detected'] = True
                self.detection_stats[obj_id] += 1
                print(f"Frame {self.frame_count}: Updated object ID:{obj_id} ({color})")
            else:
                self.object_id += 1
                self.tracked_objects[self.object_id] = {
                    'color': color,
                    'positions': [center],
                    'last_seen': self.frame_count,
                    'detected': True
                }
                self.detection_stats[self.object_id] = 1
                self.detection_history.append({
                    'id': self.object_id,
                    'color': color,
                    'first_frame': self.frame_count
                })
                print(f"Frame {self.frame_count}: New object ID:{self.object_id} ({color})")

    def process_frame(self, frame):
        """Полная обработка кадра с детекцией цветов"""
        # 1. Предварительная обработка
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        current_detections = []
        
        # 2. Детектирование цветных объектов
        for color, params in self.color_params.items():
            mask = cv2.inRange(hsv, params['lower'], params['upper'])
            
            # Морфологические операции
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if (self.min_area < area < self.max_area) and self.is_circular(cnt):
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    current_detections.append({
                        'center': center,
                        'radius': radius,
                        'color': color
                    })
                    
                    # Рисуем обнаруженные объекты
                    cv2.circle(frame, center, radius, params['color'], 2)
                    cv2.circle(frame, center, 3, params['color'], -1)
                    cv2.putText(frame, f"{color} r:{radius}", 
                               (center[0]-30, center[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, params['color'], 1)
        
        # 3. Обновление трекинга
        self._update_object_tracking(current_detections)
        
        # 4. Добавление информационной панели
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"TCP Stream: {self.server_ip}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), 
                   (frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Индикатор записи
        if self.recording:
            cv2.circle(frame, (frame.shape[1]-30, 30), 10, (0, 0, 255), -1)
        
        return frame

    def run(self):
        try:
            self.connect()
            first_frame = True
            
            while True:
                # Получаем размер кадра
                size_data = self._recv_all(4)
                if not size_data:
                    break
                
                size = struct.unpack('!I', size_data)[0]
                frame_data = self._recv_all(size)
                
                if frame_data:
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Инициализация видеозаписи при первом кадре
                        if first_frame:
                            self.init_video_writer((frame.shape[1], frame.shape[0]))
                            first_frame = False
                            
                        # Обработка кадра
                        processed_frame = self.process_frame(frame)
                        self.frame_count += 1
                        
                        # Отображение и запись
                        cv2.imshow('Drone Color Detection', processed_frame)
                        if self.recording:
                            self.video_writer.write(processed_frame)
                
                # Управление клавишами
                key = cv2.waitKey(1)
                if key == 27:  # ESC - выход
                    break
                elif key == ord(' '):  # Пробел - пауза записи
                    self.recording = not self.recording
                    print(f"Recording {'paused' if not self.recording else 'resumed'}")
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
            self.print_final_stats()

    def _recv_all(self, n):
        """Надежный прием всех данных"""
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def cleanup(self):
        """Очистка ресурсов"""
        if self.video_writer is not None:
            self.video_writer.release()
        if self.client_socket:
            self.client_socket.close()
        cv2.destroyAllWindows()

    def print_final_stats(self):
        """Вывод итоговой статистики"""
        # Фильтрация объектов по количеству обнаружений
        valid_ids = {obj_id for obj_id, n in self.detection_stats.items()
                    if n >= self.min_detections}

        # Сортировка по порядку появления
        ordered = sorted(self.detection_history, key=lambda x: x['first_frame'])

        # Подсчет по цветам
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        for obj_id in valid_ids:
            color = self.tracked_objects[obj_id]['color']
            color_counts[color] += 1

        print("\n=== Final Processing Statistics ===")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total objects detected: {len(valid_ids)}")
        print("Objects by color:")
        for c in ['red', 'green', 'blue']:
            print(f"  {c}: {color_counts[c]}")

        print("\nDetection order (first appearance):")
        order_num = 1
        for entry in ordered:
            obj_id = entry['id']
            if obj_id not in valid_ids:
                continue
            color = entry['color']
            first_frame = entry['first_frame']
            detections = self.detection_stats[obj_id]
            print(f"{order_num}. ID:{obj_id} - {color}, first frame: {first_frame}, "
                f"detections: {detections}")
            order_num += 1

        if order_num == 1:
            print("No objects passed the stability threshold.")

if __name__ == '__main__':
    detector = DroneColorDetector(server_ip='192.168.0.6')  # Укажите IP дрона
    detector.run()