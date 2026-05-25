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
            # Кулак - все пальцы опущены
            is_fist1 = count1 == 0
            is_fist2 = count2 == 0

            if is_fist1 or is_fist2:
                return "Кулак"

            # === OPEN PALM (две открытые ладони к камере) ===
            # Все пальцы подняты на обеих руках
            if count1 == 5 and count2 == 5:
                return "Открытая ладонь"

            # === TWO PALMS PARALLEL (две ладони друг к другу) ===
            # Проверяем, что ладони повёрнуты друг к другу
            # Для этого смотрим на положение больших пальцев
            thumb1_x = points1[4]['x']
            thumb2_x = points2[4]['x']
            palm1_x = points1[0]['x']  # Запястье
            palm2_x = points2[0]['x']

            # Если левая рука справа, а правая слева - ладони повёрнуты друг к другу
            if type1 == 'Left' and type2 == 'Right':
                if thumb1_x > palm1_x and thumb2_x < palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Две ладони параллельно"
            elif type1 == 'Right' and type2 == 'Left':
                if thumb1_x < palm1_x and thumb2_x > palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Две ладони параллельно"

            # === I DON'T KNOW (ладони в разные стороны) ===
            # Проверяем, что большие пальцы смотрят наружу
            if type1 == 'Left' and type2 == 'Right':
                if thumb1_x < palm1_x and thumb2_x > palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Я не знаю"
            elif type1 == 'Right' and type2 == 'Left':
                if thumb1_x > palm1_x and thumb2_x < palm2_x:
                    if count1 == 5 and count2 == 5:
                        return "Я не знаю"

            # Если две руки но не подошли под специальные жесты
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

            # === LITTLE (жест "мало") ===
            # Указательный и большой подняты, остальные опущены
            # [thumb, index, middle, ring, pinky]
            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                # Дополнительная проверка: пальцы должны быть вытянуты
                thumb_tip = points[4]
                index_tip = points[8]
                distance = self.calculate_distance(thumb_tip, index_tip)
                if distance < 80:  # Пальцы не слишком далеко друг от друга
                    return "Мало"

            # === POINTING UP (указательный палец вверх) ===
            if not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                return "Указывая вверх"



            # === THUMBS UP (большой палец вверх) ===
            if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                return "Большой палец вверх"

            # === PEACE / VICTORY (два пальца) ===
            if not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                return "Мир"

            # === OPEN PALM (открытая ладонь) ===
            if finger_count == 5:
                return "Открытая ладонь"

            # === CHERRY (вишенка - все пальцы касаются большого) ===
            thumb_tip = points[4]
            all_touching = True
            for tip_idx in [8, 12, 16, 20]:
                distance = self.calculate_distance(thumb_tip, points[tip_idx])
                if distance > 50:
                    all_touching = False
                    break

            if all_touching and finger_count > 0:
                return "Вишенка на торте"

            # === ROCK (Spiderman) - указательный и мизинец ===
            if fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
                return "Рок"

            # === OK жест ===
            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                distance = self.calculate_distance(points[4], points[8])
                if distance < 40:
                    return "OK"

            # === FACE PALM (рука-лицо) ===
            # Проверяем, что ладонь находится в верхней части кадра
            # и пальцы направлены вверх
            palm_center_y = points[0]['y']  # Запястье
            if palm_center_y < 250 and finger_count >= 3:
                # Проверяем, что пальцы направлены вверх
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

    def release(self):
        self.hands.close()