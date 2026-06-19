import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
import math


class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.7, min_tracking_confidence=0.7):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS

        # Настройка цветов для рисования
        self.landmark_color = (255, 100, 0)
        self.connection_color = (255, 200, 0)
        self.landmark_radius = 4
        self.connection_thickness = 2

    def process_frame(self, frame):
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        result = self.hands.process(rgb_frame)

        rgb_frame.flags.writeable = True
        annotated_frame = frame.copy()
        hands_landmarks = []

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks,
                                                  result.multi_handedness):

                # Рисуем соединения
                for connection in self.hand_connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]

                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(annotated_frame, start_point, end_point,
                             self.connection_color, self.connection_thickness)

                # Рисуем точки
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(annotated_frame, (x, y), self.landmark_radius,
                               self.landmark_color, -1)
                    cv2.circle(annotated_frame, (x, y), self.landmark_radius,
                               (255, 255, 255), 1)

                h_frame, w_frame, _ = frame.shape
                hand_points = [None] * 21

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    hand_points[idx] = {
                        'index': idx,
                        'x': int(landmark.x * w_frame),
                        'y': int(landmark.y * h_frame),
                        'z': float(landmark.z),
                        'normalized_x': float(landmark.x),
                        'normalized_y': float(landmark.y)
                    }

                hands_landmarks.append({
                    'hand_type': handedness.classification[0].label,
                    'confidence': float(handedness.classification[0].score),
                    'landmarks': hand_points,
                    'timestamp': datetime.now().isoformat()
                })

        return annotated_frame, hands_landmarks

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)

    def get_finger_states(self, landmarks, hand_type):
        """
        Определяет какие пальцы подняты
        Возвращает список из 5 булевых значений: [thumb, index, middle, ring, pinky]
        """
        fingers = [False, False, False, False, False]

        # Большой палец (по горизонтали)
        if hand_type == 'Right':
            fingers[0] = landmarks[4]['x'] < landmarks[3]['x']
        else:
            fingers[0] = landmarks[4]['x'] > landmarks[3]['x']

        # Указательный палец
        fingers[1] = landmarks[8]['y'] < landmarks[6]['y']

        # Средний палец
        fingers[2] = landmarks[12]['y'] < landmarks[10]['y']

        # Безымянный палец
        fingers[3] = landmarks[16]['y'] < landmarks[14]['y']

        # Мизинец
        fingers[4] = landmarks[20]['y'] < landmarks[18]['y']

        return fingers

    def get_finger_count(self, landmarks, hand_type):
        """Возвращает количество поднятых пальцев"""
        fingers = self.get_finger_states(landmarks, hand_type)
        return sum(fingers)

    def recognize_gesture(self, landmarks):
        """
        Распознавание жестов
        """
        if not landmarks:
            return "Нет рук"

        num_hands = len(landmarks)

        # ========== ЖЕСТЫ С ДВУМЯ РУКАМИ ==========
        if num_hands == 2:
            hand1 = landmarks[0]
            hand2 = landmarks[1]
            points1 = hand1['landmarks']
            points2 = hand2['landmarks']
            type1 = hand1['hand_type']
            type2 = hand2['hand_type']

            fingers1 = self.get_finger_states(points1, type1)
            fingers2 = self.get_finger_states(points2, type2)

            count1 = sum(fingers1)
            count2 = sum(fingers2)

            # === FIST (кулаки) ===
            if count1 == 0 or count2 == 0:
                return "Кулак"

            # === OPEN PALM (две открытые ладони к камере) ===
            if count1 == 5 and count2 == 5:
                return "Открытая ладонь"

            # === TWO PALMS PARALLEL ===
            thumb1_x = points1[4]['x']
            thumb2_x = points2[4]['x']
            palm1_x = points1[0]['x']
            palm2_x = points2[0]['x']

            if type1 == 'Left' and type2 == 'Right':
                if thumb1_x > palm1_x and thumb2_x < palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Две ладони параллельно"
            elif type1 == 'Right' and type2 == 'Left':
                if thumb1_x < palm1_x and thumb2_x > palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Две ладони параллельно"

            # === I DON'T KNOW ===
            if type1 == 'Left' and type2 == 'Right':
                if thumb1_x < palm1_x and thumb2_x > palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Я не знаю"
            elif type1 == 'Right' and type2 == 'Left':
                if thumb1_x > palm1_x and thumb2_x < palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Я не знаю"

            return "Две руки"

        # ========== ЖЕСТЫ С ОДНОЙ РУКОЙ ==========
        if num_hands == 1:
            hand = landmarks[0]
            points = hand['landmarks']
            hand_type = hand['hand_type']

            fingers = self.get_finger_states(points, hand_type)
            finger_count = sum(fingers)

            # === FIST (кулак) ===
            if finger_count == 0:
                return "Кулак"

            # === CHERRY (вишенка на торте) ===
            # Все пальцы согнуты и касаются большого пальца, образуя форму треугольника или щипка
            if self.is_cherry_gesture(points, hand_type):
                return "Вишенка на торте"

            # === LITTLE (жест "мало") ===
            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                thumb_tip = points[4]
                index_tip = points[8]
                distance = self.calculate_distance(thumb_tip, index_tip)
                if distance < 80:
                    return "Мало"

            # === POINTING UP ===
            if not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                return "Указывая вверх"

            # === THUMBS UP ===
            if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                return "Большой палец вверх"

            # === PEACE / VICTORY ===
            if not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                return "Мир"

            # === OPEN PALM ===
            if finger_count == 5:
                return "Открытая ладонь"

            # === ROCK ===
            if fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
                return "Рок"

            # === OK жест ===
            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                distance = self.calculate_distance(points[4], points[8])
                if distance < 40:
                    return "OK"

            # === FACE PALM ===
            palm_center_y = points[0]['y']
            if palm_center_y < 250 and finger_count >= 3:
                index_up = points[8]['y'] < points[6]['y']
                middle_up = points[12]['y'] < points[10]['y']
                if index_up and middle_up:
                    return "Фейс-палм"

            # По умолчанию - показываем количество пальцев
            if finger_count == 1:
                if fingers[0]:
                    return "Только большой палец"
                elif fingers[1]:
                    return "Только указательный"
                else:
                    return "Один палец"
            elif finger_count == 2:
                return "Два пальца"
            elif finger_count == 3:
                return "Три пальца"
            elif finger_count == 4:
                return "Четыре пальца"
            else:
                return f"{finger_count} Палец(ев)"

        return "Неизвестно"

    def is_cherry_gesture(self, landmarks, hand_type):
        """
        Проверка жеста "Вишенка на торте" - все пальцы приложены к большому,
        образуя форму, похожую на треугольник или щипок
        """
        points = landmarks

        # Кончик большого пальца
        thumb_tip = points[4]

        # Проверяем расстояние от каждого пальца до большого
        finger_tips = [8, 12, 16, 20]  # указательный, средний, безымянный, мизинец

        # Считаем количество пальцев, которые близко к большому
        close_fingers = 0
        max_distance = 55  # Максимальное расстояние для касания

        for tip_idx in finger_tips:
            tip = points[tip_idx]
            distance = self.calculate_distance(tip, thumb_tip)
            if distance < max_distance:
                close_fingers += 1

        # Для жеста "вишенка" нужно минимум 3 пальца, касающихся большого
        if close_fingers >= 3:
            return True

        # Альтернативная проверка: все пальцы согнуты и близко к центру ладони
        # Находим центр между кончиками пальцев
        center_x = sum(points[tip_idx]['x'] for tip_idx in finger_tips) / 4
        center_y = sum(points[tip_idx]['y'] for tip_idx in finger_tips) / 4

        # Расстояние от центра до большого пальца
        center_to_thumb = self.calculate_distance(thumb_tip, {'x': center_x, 'y': center_y, 'z': 0})

        # Если все пальцы собраны в центре и близко к большому
        if center_to_thumb < 60:
            # Проверяем разброс пальцев (они должны быть близко друг к другу)
            max_x = max(points[tip_idx]['x'] for tip_idx in finger_tips)
            min_x = min(points[tip_idx]['x'] for tip_idx in finger_tips)
            max_y = max(points[tip_idx]['y'] for tip_idx in finger_tips)
            min_y = min(points[tip_idx]['y'] for tip_idx in finger_tips)

            # Если пальцы собраны в кучку (разброс небольшой)
            if (max_x - min_x) < 80 and (max_y - min_y) < 80:
                return True

        # Проверка на жест "щепотка" (только указательный и большой)
        index_tip = points[8]
        index_distance = self.calculate_distance(index_tip, thumb_tip)

        middle_tip = points[12]
        middle_distance = self.calculate_distance(middle_tip, thumb_tip)

        # Если указательный и средний близко к большому, а остальные тоже близко
        if index_distance < 40 and middle_distance < 50:
            # Проверяем остальные пальцы
            ring_tip = points[16]
            pinky_tip = points[20]
            ring_distance = self.calculate_distance(ring_tip, thumb_tip)
            pinky_distance = self.calculate_distance(pinky_tip, thumb_tip)

            if ring_distance < 60 and pinky_distance < 60:
                return True

        return False

    def release(self):
        self.hands.close()