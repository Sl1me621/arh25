#!/usr/bin/env python3
import cv2
import socket
import struct
import numpy as np

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.0.6', 5000))  # Замените на реальный IP дрона
    
    try:
        while True:
            # Получаем размер изображения (4 байта)
            size_data = client_socket.recv(4)
            if not size_data:
                break
                
            size = struct.unpack('!I', size_data)[0]
            data = b''
            
            # Получаем само изображение
            while len(data) < size:
                packet = client_socket.recv(size - len(data))
                if not packet:
                    break
                data += packet
                
            # Декодируем и отображаем
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imshow('Drone Camera Stream', img)
                
            if cv2.waitKey(1) == 27:  # ESC для выхода
                break
                
    finally:
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()