#!/usr/bin/env python3
import cv2
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

class VideoFileProcessor:
    def __init__(self):
        # Параметры цветового детектирования (HSV)
        self.detection_history = []  # порядок первого появления: [{id, color, first_frame}]
        self.reappear_gap = 150  # после такого числа кадров "старый" объект считается новым

        self.color_params = {
            'red': {
                'lower': np.array([132, 131, 102]),
                'upper': np.array([179, 255, 255]),
                'color': (0, 0, 255)  # Красный в BGR
            },
            'green': {
                # 'lower': np.array([69, 119, 210]),
                # 'upper': np.array([83, 200, 255]),
                # 'color': (0, 255, 0)  # Зеленый в BGR
                'lower': np.array([69, 104, 15]),
                    'upper': np.array([77, 209, 255]),
                    'color': (0, 0, 255)  # Красный в BGR
            },
            'blue': {
                # 'lower': np.array([64, 22, 165]),
                # 'upper': np.array([95, 97, 255]),
                # 'color': (255, 0, 0)  # Синий в BGR
                'lower': np.array([65, 42, 160]),
                'upper': np.array([113, 155, 255]),
                'color': (0, 0, 255)  # Красный в BGR

            }
        }
        
        # Параметры фильтрации круглых объектов
        self.min_area = 200       # Минимальная площадь объекта
        self.max_area = 1000      # Максимальная площадь объекта
        self.circularity_thresh = 0.7  # Порог округлости
        
        # Параметры трекинга объектов
        self.min_detections = 15    # Минимальное количество обнаружений для учета объекта
        self.max_distance =150     # Максимальное расстояние между кадрами для трекинга
        
        # Статистика
        self.object_id = 0
        self.tracked_objects = {}  # {id: {'color': str, 'positions': list, 'last_seen': int}}
        self.detection_stats = defaultdict(int)  # Для подсчета временных обнаружений
        
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_video_file(self, input_file, output_file=None):
        """Обработка видеофайла"""
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_file}")
            return
        
        # Получаем параметры видео для записи
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, 
                                             (frame_width, frame_height))
            self.recording = True
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            
            # Отображение результата
            cv2.imshow('Video Processing', processed_frame)
            
            # Запись в файл если нужно
            if self.recording:
                self.video_writer.write(processed_frame)
            
            # Выход по ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            self.frame_count += 1
        
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Анализ и вывод статистики
        self._analyze_and_print_stats()
    
    def _analyze_and_print_stats(self):
        """Итоговый вывод: объекты по порядку первого появления."""
        # Фильтруем объекты по устойчивости (как и раньше)
        valid_ids = {obj_id for obj_id, n in self.detection_stats.items()
                    if n >= self.min_detections}

        # Сортируем историю по кадру первого появления
        ordered = sorted(self.detection_history, key=lambda x: x['first_frame'])

        # Счётчики по цветам (по валидным)
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        for obj_id in valid_ids:
            color = self.tracked_objects[obj_id]['color']
            color_counts[color] += 1

        print("\n=== Результаты обработки видео ===")
        print(f"Всего кадров обработано: {self.frame_count}")
        print(f"Всего объектов обнаружено: {len(valid_ids)}")
        print("По цветам:")
        for c in ['red', 'green', 'blue']:
            print(f"  {c}: {color_counts[c]}")

        # Порядок нахождения (только валидные объекты)
        print("\nПорядок обнаружения объектов (от начала видео):")
        order_num = 1
        for entry in ordered:
            obj_id = entry['id']
            if obj_id not in valid_ids:
                continue
            color = entry['color']
            first_frame = entry['first_frame']
            detections = self.detection_stats[obj_id]
            print(f"{order_num}. ID:{obj_id} — {color}, первый кадр: {first_frame}, "
                f"всего детекций: {detections}")
            order_num += 1

        if order_num == 1:
            print("Нет объектов, прошедших порог устойчивости.")

    
    def _get_nearest_object(self, point, color):
        min_dist = float('inf')
        nearest_id = None
        
        for obj_id, obj_info in self.tracked_objects.items():
            if obj_info['color'] != color:
                continue

            # Проверка: если объект слишком давно не видели — не использовать
            if self.frame_count - obj_info['last_seen'] > self.reappear_gap:
                continue

            last_pos = obj_info['positions'][-1]
            dist = np.linalg.norm(np.array(point) - np.array(last_pos))
            
            if dist < self.max_distance and dist < min_dist:
                min_dist = dist
                nearest_id = obj_id
    
        return nearest_id if min_dist < self.max_distance else None

    
    def is_circular(self, cnt):
        """Проверка, является ли контур круглым"""
        area = cv2.contourArea(cnt)
        if area == 0:
            return False
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return False
            
        # Вычисляем коэффициент округлости
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > self.circularity_thresh
    
    def process_frame(self, frame):
        """Обработка одного кадра с трекингом объектов"""
        # 1. Предварительная обработка
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 2. Детектирование цветных объектов
        current_detections = []
        
        for color, params in self.color_params.items():
            mask = cv2.inRange(hsv, params['lower'], params['upper'])
            
            # Улучшенные морфологические операции
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Фильтрация по размеру и форме
                if (self.min_area < area < self.max_area) and self.is_circular(cnt):
                    # Получаем параметры окружности
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    # Добавляем в текущие обнаружения
                    current_detections.append({
                        'center': center,
                        'radius': radius,
                        'color': color
                    })
                    
                    # Рисуем окружность и центр
                    cv2.circle(frame, center, radius, params['color'], 2)
                    cv2.circle(frame, center, 3, params['color'], -1)
                    
                    # Подпись с информацией
                    cv2.putText(frame, f"{color} r:{radius}", (center[0]-30, center[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, params['color'], 1)
        
        # 3. Трекинг объектов между кадрами
        self._update_object_tracking(current_detections)
        
        # 4. Добавление информационной панели
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _update_object_tracking(self, current_detections):
        """Обновление трекинга объектов"""
        # Помечаем все объекты как необнаруженные в этом кадре
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['detected'] = False
        
        # Обработка текущих обнаружений
        for detection in current_detections:
            center = detection['center']
            color = detection['color']
            
            # Ищем ближайший объект того же цвета
            obj_id = self._get_nearest_object(center, color)
            
            if obj_id is not None:
                # Обновляем существующий объект
                self.tracked_objects[obj_id]['positions'].append(center)
                self.tracked_objects[obj_id]['last_seen'] = self.frame_count
                self.tracked_objects[obj_id]['detected'] = True
                self.detection_stats[obj_id] += 1

                # выводим в терминал
                print(f"Кадр {self.frame_count}: Обновлен объект ID:{obj_id} ({color})")
            else:
                # Создаем новый объект
                self.object_id += 1
                self.tracked_objects[self.object_id] = {
                    'color': color,
                    'positions': [center],
                    'last_seen': self.frame_count,
                    'detected': True
                }
                self.detection_stats[self.object_id] = 1

                # фиксируем порядок первого появления
                self.detection_history.append({
                    'id': self.object_id,
                    'color': color,
                    'first_frame': self.frame_count
                })

                # выводим в терминал
                print(f"Кадр {self.frame_count}: Новый объект ID:{self.object_id} ({color})")

        
        # Удаляем объекты, которые давно не видели (опционально)
        # to_delete = [obj_id for obj_id, obj in self.tracked_objects.items() 
        #             if self.frame_count - obj['last_seen'] > 10]
        # for obj_id in to_delete:
        #     del self.tracked_objects[obj_id]
        #     del self.detection_stats[obj_id]

if __name__ == '__main__':
    processor = VideoFileProcessor()
    
    # Укажите путь к входному видеофайлу
    input_video = "drone_processed_20250816_170950.avi"
    output_video = "processed_output.avi"  # или None если не нужно сохранять
    
    processor.process_video_file(input_video, output_video)