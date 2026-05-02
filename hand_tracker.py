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
        self.simple_drawing = True

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

                if self.simple_drawing:
                    for connection in self.hand_connections:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        start = hand_landmarks.landmark[start_idx]
                        end = hand_landmarks.landmark[end_idx]

                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
                else:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.hand_connections
                    )

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

    def calculate_angle(self, p1, p2, p3):
        try:
            a = self.calculate_distance(p1, p2)
            b = self.calculate_distance(p2, p3)
            c = self.calculate_distance(p1, p3)

            if a * b == 0:
                return 180

            cos_value = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

            if cos_value > 1:
                cos_value = 1
            elif cos_value < -1:
                cos_value = -1

            angle = math.acos(cos_value)
            return math.degrees(angle)
        except:
            return 180

    def get_finger_state_accurate(self, landmarks, hand_type):
        fingers = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }

        if hand_type == 'Right':
            fingers['thumb'] = landmarks[4]['x'] < landmarks[3]['x']
        else:
            fingers['thumb'] = landmarks[4]['x'] > landmarks[3]['x']

        fingers['index'] = landmarks[8]['y'] < landmarks[6]['y']
        fingers['middle'] = landmarks[12]['y'] < landmarks[10]['y']
        fingers['ring'] = landmarks[16]['y'] < landmarks[14]['y']
        fingers['pinky'] = landmarks[20]['y'] < landmarks[18]['y']

        return fingers

    def is_pinch_gesture(self, landmarks):
        """Проверяет жест 'мало' (указательный над большим, остальные загнуты).."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        # Проверяем, что указательный и большой палец близко (пинч)
        distance = self.calculate_distance(thumb_tip, index_tip)

        # Указательный должен быть выше большого
        is_index_above_thumb = index_tip['y'] < thumb_tip['y']

        # Остальные пальцы должны быть загнуты
        fingers = self.get_finger_state_accurate(landmarks, 'Right')
        others_closed = not fingers['middle'] and not fingers['ring'] and not fingers['pinky']

        return distance < 40 and is_index_above_thumb and others_closed

    def get_palm_orientation(self, landmarks, hand_type):
        """
        Определяет ориентацию ладони
        Возвращает:
        - 'facing_camera' - ладонь повернута к камере
        - 'sideways_left' - ладонь повернута боком влево
        - 'sideways_right' - ладонь повернута боком вправо
        """
        # Берем ключевые точки: запястье (0), основание указательного (5), основание мизинца (17)
        wrist = landmarks[0]
        index_base = landmarks[5]
        pinky_base = landmarks[17]

        # Вычисляем ширину ладони в пикселях
        palm_width = abs(index_base['x'] - pinky_base['x'])
        palm_height = abs(index_base['y'] - pinky_base['y'])

        # Если ширина ладони больше высоты, значит ладонь повернута к камере
        if palm_width > palm_height * 0.9:
            return 'facing_camera'
        else:
            # Определяем, в какую сторону повернута ладонь
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]

            # Для правой руки: если большой палец справа от мизинца - ладонь смотрит влево
            if hand_type == 'Right':
                if thumb_tip['x'] > pinky_tip['x']:
                    return 'sideways_left'  # Ладонь смотрит влево
                else:
                    return 'sideways_right'  # Ладонь смотрит вправо
            else:  # Left hand
                if thumb_tip['x'] < pinky_tip['x']:
                    return 'sideways_right'  # Ладонь смотрит вправо
                else:
                    return 'sideways_left'  # Ладонь смотрит влево

    def are_palms_facing_each_other_sideways(self, hands_landmarks):
        """Проверяет, что две ладони повернуты боком друг к другу (параллельно)"""
        if len(hands_landmarks) != 2:
            return False

        hand1 = hands_landmarks[0]
        hand2 = hands_landmarks[1]

        points1 = hand1['landmarks']
        points2 = hand2['landmarks']
        hand_type1 = hand1['hand_type']
        hand_type2 = hand2['hand_type']

        # Проверяем, что обе руки открыты (все пальцы выпрямлены)
        fingers1 = self.get_finger_state_accurate(points1, hand_type1)
        fingers2 = self.get_finger_state_accurate(points2, hand_type2)

        if not (all(fingers1.values()) and all(fingers2.values())):
            return False

        # Получаем ориентацию каждой ладони
        orientation1 = self.get_palm_orientation(points1, hand_type1)
        orientation2 = self.get_palm_orientation(points2, hand_type2)

        # Обе ладони должны быть повернуты боком (не к камере)
        if orientation1 == 'facing_camera' or orientation2 == 'facing_camera':
            return False

        # Ладони должны смотреть друг на друга
        # Если левая ладонь смотрит вправо, а правая влево - они смотрят друг на друга
        if hand_type1 == 'Left' and hand_type2 == 'Right':
            # Левая рука должна смотреть вправо, правая - влево
            return orientation1 == 'sideways_right' and orientation2 == 'sideways_left'
        elif hand_type1 == 'Right' and hand_type2 == 'Left':
            # Правая рука должна смотреть влево, левая - вправо
            return orientation1 == 'sideways_left' and orientation2 == 'sideways_right'

        return False

    def are_palms_facing_opposite_directions(self, hands_landmarks):
        """Проверяет, что две ладони повернуты в разные стороны (жест 'не знаю')"""
        if len(hands_landmarks) != 2:
            return False

        hand1 = hands_landmarks[0]
        hand2 = hands_landmarks[1]

        points1 = hand1['landmarks']
        points2 = hand2['landmarks']
        hand_type1 = hand1['hand_type']
        hand_type2 = hand2['hand_type']

        # Проверяем, что обе руки открыты
        fingers1 = self.get_finger_state_accurate(points1, hand_type1)
        fingers2 = self.get_finger_state_accurate(points2, hand_type2)

        if not (all(fingers1.values()) and all(fingers2.values())):
            return False

        # Получаем ориентацию каждой ладони
        orientation1 = self.get_palm_orientation(points1, hand_type1)
        orientation2 = self.get_palm_orientation(points2, hand_type2)

        # Обе ладони должны быть повернуты боком
        if orientation1 == 'facing_camera' or orientation2 == 'facing_camera':
            return False

        # Ладони должны смотреть в разные стороны (обе влево или обе вправо)
        return orientation1 == orientation2

    def are_palms_facing_camera(self, hands_landmarks):
        """Проверяет, что все ладони повернуты к камере"""
        for hand in hands_landmarks:
            points = hand['landmarks']
            hand_type = hand['hand_type']

            # Проверяем, что рука открыта
            fingers = self.get_finger_state_accurate(points, hand_type)
            if not all(fingers.values()):
                return False

            # Проверяем ориентацию ладони
            orientation = self.get_palm_orientation(points, hand_type)
            if orientation != 'facing_camera':
                return False

        return True

    def is_cherry_gesture(self, landmarks, hand_type):
        """Проверяет жест 'вишенка' - все пальцы касаются большого"""
        thumb_tip = landmarks[4]

        # Проверяем, что все кончики пальцев близко к большому пальцу
        finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
        all_touching_thumb = True

        for tip_idx in finger_tips:
            tip = landmarks[tip_idx]
            distance = self.calculate_distance(thumb_tip, tip)
            if distance > 60:  # Порог касания
                all_touching_thumb = False
                break

        # Также проверяем, что средние фаланги не слишком высоко
        if all_touching_thumb:
            # Дополнительная проверка: кончики должны быть ниже кончиков пальцев
            for tip_idx in finger_tips:
                tip = landmarks[tip_idx]
                dip_idx = tip_idx - 2
                dip = landmarks[dip_idx]
                if tip['y'] < dip['y']:  # Палец выпрямлен
                    all_touching_thumb = False
                    break

        return all_touching_thumb

    def recognize_gesture(self, landmarks):
        if not landmarks:
            return "No hand"

        num_hands = len(landmarks)

        # Жест "вишенка" (1 рука)
        if num_hands == 1:
            hand = landmarks[0]
            points = hand['landmarks']
            hand_type = hand['hand_type']

            # Проверяем жест "вишенка"
            if self.is_cherry_gesture(points, hand_type):
                return "Cherry"

        # Жест "мало" (1 рука)
        if num_hands == 1:
            hand = landmarks[0]
            points = hand['landmarks']
            if self.is_pinch_gesture(points):
                return "Little"

        # === ПРОВЕРКА ДЛЯ ДВУХ РУК ===
        if num_hands == 2:
            # Проверяем жест "ладони повернуты боком друг к другу" (у лица)
            if self.are_palms_facing_each_other_sideways(landmarks):
                return "Two Palms Parallel"

            # Проверяем жест "ладони повернуты в разные стороны" (не знаю)
            if self.are_palms_facing_opposite_directions(landmarks):
                return "I Don't Know"

            # Проверяем, что обе руки - открытые ладони, повернутые к камере
            if self.are_palms_facing_camera(landmarks):
                return "Open Palm"

            return "Two Hands"

        # === ДЛЯ ОДНОЙ РУКИ ===
        if num_hands == 1:
            hand = landmarks[0]
            points = hand['landmarks']
            hand_type = hand['hand_type']

            fingers = self.get_finger_state_accurate(points, hand_type)
            finger_count = sum(fingers.values())

            thumb_tip = points[4]
            index_tip = points[8]
            middle_tip = points[12]

            thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
            index_middle_dist = self.calculate_distance(index_tip, middle_tip)

            # Pointing Up (указательный палец вверх)
            if not fingers['thumb'] and fingers['index'] and not fingers['middle'] and not fingers['ring'] and not \
            fingers['pinky']:
                return "Pointing Up"

            # Thumbs Up (большой палец вверх)
            if fingers['thumb'] and not fingers['index'] and not fingers['middle'] and not fingers['ring'] and not \
            fingers['pinky']:
                return "Thumbs Up"

            # Peace / Victory (V)
            if not fingers['thumb'] and fingers['index'] and fingers['middle'] and not fingers['ring'] and not fingers[
                'pinky']:
                return "Peace"

            # Open Palm (одна открытая ладонь, повернутая к камере)
            if finger_count == 5:
                palm_check = True
                finger_indices = [4, 8, 12, 16, 20]
                dip_indices = [3, 6, 10, 14, 18]
                for tip, dip in zip(finger_indices, dip_indices):
                    if points[tip]['y'] > points[dip]['y']:
                        palm_check = False
                        break
                if palm_check:
                    # Дополнительно проверяем, что ладонь повернута к камере
                    orientation = self.get_palm_orientation(points, hand_type)
                    if orientation == 'facing_camera':
                        return "Open Palm"

            if finger_count == 0:
                fist_check = True
                finger_indices = [4, 8, 12, 16, 20]
                dip_indices = [3, 6, 10, 14, 18]
                for tip, dip in zip(finger_indices, dip_indices):
                    if points[tip]['y'] < points[dip]['y']:
                        fist_check = False
                        break
                if fist_check:
                    return "Fist"

            if fingers['thumb'] and fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers[
                'pinky']:
                if thumb_index_dist < 40:
                    return "OK"

            if fingers['index'] and not fingers['middle'] and not fingers['ring'] and fingers['pinky']:
                return "Rock (Spiderman)"

            if fingers['index'] and fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
                if index_middle_dist < 30:
                    return "Crossed Fingers"

            if fingers['thumb'] and fingers['index'] and fingers['middle'] and not fingers['ring'] and not fingers[
                'pinky']:
                return "Three (Chinese)"

            if fingers['thumb'] and fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers[
                'pinky']:
                return "Two (Chinese)"

            if finger_count == 1:
                if fingers['thumb']:
                    return "Thumb Only"
                elif fingers['index']:
                    return "Index Only"
                elif fingers['middle']:
                    return "Middle Only"
                elif fingers['ring']:
                    return "Ring Only"
                elif fingers['pinky']:
                    return "Pinky Only"

            if finger_count == 2:
                if fingers['index'] and fingers['middle']:
                    return "Two Fingers"
                elif fingers['index'] and fingers['ring']:
                    return "Two Fingers"
                elif fingers['index'] and fingers['pinky']:
                    return "Two Fingers"
                elif fingers['middle'] and fingers['ring']:
                    return "Two Fingers"

            if finger_count == 3:
                if not fingers['thumb'] and fingers['index'] and fingers['middle'] and fingers['ring']:
                    return "Three Fingers"
                elif fingers['thumb'] and fingers['index'] and fingers['middle']:
                    return "Three Fingers"

            if finger_count == 4:
                if not fingers['thumb']:
                    return "Four Fingers"
                elif not fingers['pinky']:
                    return "Four Fingers"

            return f"{finger_count} Finger(s)"

        return "Unknown"

    def save_landmarks_to_file(self, landmarks_data, filename='data/hand_landmarks.json'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        existing_data.append({
            'timestamp': datetime.now().isoformat(),
            'hands': landmarks_data
        })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        return len(existing_data) - 1

    def release(self):
        self.hands.close()