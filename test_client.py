#!/usr/bin/env python3
import cv2
import socket
import struct
import numpy as np
import time
from datetime import datetime

class VideoProcessor:
    def __init__(self, server_ip='192.168.1.100', port=5000):
        self.server_ip = server_ip
        self.port = port
        self.client_socket = None
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Параметры цветового детектирования (HSV)
        self.color_params = {
            'red': {
                'lower': np.array([133, 154, 0]),
                'upper': np.array([179, 255, 255]),
                'color': (0, 0, 255)  # Красный в BGR
            },
            'green': {
                    'lower': np.array([np.int64(0), np.int64(116), np.int64(146)]),
                    'upper': np.array([np.int64(88), np.int64(229), np.int64(203)]),
                    'color': (0, 255, 0)
            },
            'blue': {
                'lower': np.array([np.int64(0), np.int64(69), np.int64(165)]),
                'upper': np.array([np.int64(104), np.int64(128), np.int64(255)]),
                'color': (255, 0, 0)
            }
        }

    def connect(self):
        """Установка соединения с сервером"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.port))
        print(f"Connected to {self.server_ip}:{self.port}")

    def init_video_writer(self, frame):
        """Инициализация записи видео"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drone_footage_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        frame_size = (frame.shape[1], frame.shape[0])
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.recording = True
        print(f"Recording started: {filename}")

    def process_frame(self, frame):
        """Полная обработка кадра"""
        # 1. Предварительная обработка
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 2. Детектирование цветных объектов
        for color, params in self.color_params.items():
            mask = cv2.inRange(hsv, params['lower'], params['upper'])
            
            # Морфологические операции для улучшения качества маски
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:  # Фильтр по размеру
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), params['color'], 2)
                    
                    # Вычисление центра объекта
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # cv2.circle(frame, (cX, cY), 4, params['color'], -1)
                    
                    cv2.putText(frame, f"{color} ({area}px)", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 3. Добавление информационной панели
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. Запись кадра
        if self.recording:
            self.video_writer.write(frame)
            # cv2.circle(frame, (frame.shape[1]-30, 30), 10, (0, 0, 255), -1)
        
        return frame

    def run(self):
        try:
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
                        if not self.recording:
                            self.init_video_writer(frame)
                            
                        processed_frame = self.process_frame(frame)
                        cv2.imshow('Drone Video Processing', processed_frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC - выход
                    break
                elif key == ord(' '):  # Пробел - пауза записи
                    self.recording = not self.recording
                    print(f"Recording {'paused' if not self.recording else 'resumed'}")
                    
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            self.client_socket.close()
            cv2.destroyAllWindows()
            print("Connection closed")

    def _recv_all(self, n):
        """Надежное получение всех данных"""
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

if __name__ == '__main__':
    processor = VideoProcessor(server_ip='192.168.0.6')  # Укажите IP дрона
    processor.connect()
    processor.run()