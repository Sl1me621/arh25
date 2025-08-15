import cv2
import numpy as np

def get_color_thresholds_from_image(image_path, color_name='red'):
    """
    Определяет HSV-пороги для заданного цвета на изображении.
    :param image_path: Путь к изображению
    :param color_name: Название цвета (для вывода)
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    # Конвертируем в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создаем окно для настройки порогов
    cv2.namedWindow('Color Threshold')

    # Создаем trackbars
    def nothing(x):
        pass

    cv2.createTrackbar('H Lower', 'Color Threshold', 0, 179, nothing)
    cv2.createTrackbar('H Upper', 'Color Threshold', 179, 179, nothing)
    cv2.createTrackbar('S Lower', 'Color Threshold', 0, 255, nothing)
    cv2.createTrackbar('S Upper', 'Color Threshold', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'Color Threshold', 0, 255, nothing)
    cv2.createTrackbar('V Upper', 'Color Threshold', 255, 255, nothing)

    # Установка начальных значений для красного
    cv2.setTrackbarPos('H Lower', 'Color Threshold', 0)
    cv2.setTrackbarPos('H Upper', 'Color Threshold', 10)
    cv2.setTrackbarPos('S Lower', 'Color Threshold', 120)
    cv2.setTrackbarPos('S Upper', 'Color Threshold', 255)
    cv2.setTrackbarPos('V Lower', 'Color Threshold', 70)
    cv2.setTrackbarPos('V Upper', 'Color Threshold', 255)

    while True:
        # Получаем текущие значения trackbars
        h_low = cv2.getTrackbarPos('H Lower', 'Color Threshold')
        h_high = cv2.getTrackbarPos('H Upper', 'Color Threshold')
        s_low = cv2.getTrackbarPos('S Lower', 'Color Threshold')
        s_high = cv2.getTrackbarPos('S Upper', 'Color Threshold')
        v_low = cv2.getTrackbarPos('V Lower', 'Color Threshold')
        v_high = cv2.getTrackbarPos('V Upper', 'Color Threshold')

        # Создаем маску и применяем ее
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Показываем результат
        cv2.imshow('Original', frame)
        cv2.imshow('Color Threshold', result)

        # Выход по нажатию 'q' или ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

    # Вывод результата
    print(f"'{color_name}': {{")
    print(f"    'lower': np.array([{h_low}, {s_low}, {v_low}]),")
    print(f"    'upper': np.array([{h_high}, {s_high}, {v_high}]),")
    print(f"    'color': (0, 0, 255)  # Красный в BGR")
    print("}")

if __name__ == "__main__":
    image_path = "C:/Users/User/Desktop/image1.png"  # Укажите путь к изображению
    get_color_thresholds_from_image(image_path, 'red')