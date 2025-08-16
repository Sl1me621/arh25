#!/usr/bin/env python3
import cv2
import socket
import struct
import numpy as np
import json

class DetectionClient:
    def __init__(self, server_ip='192.168.0.6', port=5000):
        self.server_ip = server_ip
        self.port = port
        self.client_socket = None

    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.port))
        print(f"Connected to {self.server_ip}:{self.port}")

    def receive_frame(self):
        # Получаем размер JSON данных
        json_size_data = self.client_socket.recv(4)
        if not json_size_data:
            return None, None
            
        json_size = struct.unpack('!I', json_size_data)[0]
        
        # Получаем JSON данные
        json_data = self._recv_all(json_size)
        if not json_data:
            return None, None
            
        data = json.loads(json_data.decode('utf-8'))
        
        # Получаем изображение
        img_data = self._recv_all(data['image_size'])
        if not img_data:
            return None, None
            
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        return img, data['objects']

    def _recv_all(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def display_objects_info(self, frame, objects):
        """Отображаем информацию об объектах"""
        for i, obj in enumerate(objects):
            cv2.putText(frame, f"{i+1}. {obj['color']} ({obj['area']}px)", 
                       (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)

    def run(self):
        try:
            self.connect()
            
            while True:
                frame, objects = self.receive_frame()
                if frame is None:
                    break
                    
                if objects:
                    self.display_objects_info(frame, objects)
                
                cv2.imshow('Processed Drone Feed', frame)
                
                if cv2.waitKey(1) == 27:  # ESC для выхода
                    break
                    
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if self.client_socket:
                self.client_socket.close()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    client = DetectionClient()
    client.run()