import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime


class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Хранилище для точек
        self.landmarks_data = []

    def process_frame(self, frame):
        """Обрабатывает кадр и возвращает размеченное изображение и координаты точек"""

        # Конвертируем BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        # Создаём копию кадра для отрисовки
        annotated_frame = frame.copy()

        # Список для хранения координат всех рук
        hands_landmarks = []

        # Если есть руки - рисуем точки и собираем координаты
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks,
                                                  result.multi_handedness):
                # Рисуем точки и соединения
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Собираем координаты точек (21 точка для каждой руки)
                h, w, _ = frame.shape
                hand_points = []

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z  # глубина

                    hand_points.append({
                        'index': idx,
                        'x': x,
                        'y': y,
                        'z': float(z),
                        'normalized_x': float(landmark.x),
                        'normalized_y': float(landmark.y)
                    })

                    # Рисуем номер точки
                    cv2.putText(annotated_frame, str(idx), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # Добавляем информацию о руке
                hands_landmarks.append({
                    'hand_type': handedness.classification[0].label,  # Left или Right
                    'confidence': float(handedness.classification[0].score),
                    'landmarks': hand_points,
                    'timestamp': datetime.now().isoformat()
                })

        return annotated_frame, hands_landmarks

    def save_landmarks_to_file(self, landmarks_data, filename='data/hand_landmarks.json'):
        """Сохраняет массив точек в JSON файл"""

        # Создаём директорию если её нет
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Загружаем существующие данные (если есть)
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        # Добавляем новые данные
        existing_data.append({
            'timestamp': datetime.now().isoformat(),
            'hands': landmarks_data
        })

        # Сохраняем обратно
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        return len(existing_data) - 1  # возвращаем индекс записи

    def recognize_gesture(self, landmarks):
        """Распознаёт простые жесты на основе координат точек"""
        if not landmarks:
            return "Нет руки"

        # Берем первую руку
        hand = landmarks[0]
        points = hand['landmarks']

        # Индексы ключевых точек
        thumb_tip = points[4]  # кончик большого пальца
        thumb_ip = points[3]  # сустав большого пальца
        index_tip = points[8]  # кончик указательного
        index_pip = points[6]  # средний сустав указательного
        middle_tip = points[12]  # кончик среднего
        ring_tip = points[16]  # кончик безымянного
        pinky_tip = points[20]  # кончик мизинца

        # Проверка поднятых пальцев (по Y координате)
        fingers_up = []

        # Большой палец (по X координате, зависит от руки)
        if hand['hand_type'] == 'Right':
            thumb_up = thumb_tip['x'] > thumb_ip['x']
        else:
            thumb_up = thumb_tip['x'] < thumb_ip['x']
        fingers_up.append(thumb_up)

        # Остальные пальцы
        fingers_up.append(index_tip['y'] < index_pip['y'])  # указательный
        fingers_up.append(middle_tip['y'] < points[10]['y'])  # средний
        fingers_up.append(ring_tip['y'] < points[14]['y'])  # безымянный
        fingers_up.append(pinky_tip['y'] < points[18]['y'])  # мизинец

        count_up = sum(fingers_up)

        # Определяем жесты
        if count_up == 0:
            return "✊ Кулак"
        elif count_up == 1 and fingers_up[1]:
            return "☝️ Указательный палец"
        elif count_up == 2 and fingers_up[1] and fingers_up[2]:
            return "✌️ Мир (V)"
        elif count_up == 3 and fingers_up[1] and fingers_up[2] and fingers_up[3]:
            return "🤟 Три пальца"
        elif count_up == 4:
            return "🖖 Четыре пальца"
        elif count_up == 5:
            return "🖐️ Открытая ладонь"
        elif fingers_up[0] and not any(fingers_up[1:]):
            return "👍 Класс (палец вверх)"
        else:
            return f"{count_up} пальца(ев)"

    def release(self):
        """Освобождает ресурсы"""
        self.hands.close()