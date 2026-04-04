from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import base64
import json
import os
from hand_tracker import HandTracker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Инициализируем трекер
tracker = HandTracker()

# Глобальная переменная для хранения последних данных
current_landmarks = []
current_gesture = "Ожидание"


def generate_frames():
    """Генератор кадров для видеопотока"""
    global current_landmarks, current_gesture

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Обрабатываем кадр
        annotated_frame, landmarks = tracker.process_frame(frame)

        # Обновляем глобальные данные
        current_landmarks = landmarks

        # Распознаём жест
        if landmarks:
            current_gesture = tracker.recognize_gesture(landmarks)
        else:
            current_gesture = "Нет руки"

        # Добавляем информацию о жесте на кадр
        cv2.putText(annotated_frame, f"Gesture: {current_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Hands: {len(landmarks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Кодируем в JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Видеопоток с распознаванием"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/landmarks')
def get_landmarks():
    """API для получения текущих координат точек"""
    global current_landmarks, current_gesture
    return jsonify({
        'hands': current_landmarks,
        'gesture': current_gesture,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })


@app.route('/api/save_landmarks', methods=['POST'])
def save_landmarks():
    """Сохраняет текущие точки в файл"""
    global current_landmarks

    if current_landmarks:
        index = tracker.save_landmarks_to_file(current_landmarks)
        return jsonify({
            'success': True,
            'message': f'Данные сохранены (запись #{index})',
            'record_id': index
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Нет данных для сохранения'
        }), 400


@app.route('/api/save_custom', methods=['POST'])
def save_custom_landmarks():
    """Сохраняет произвольные данные точек"""
    data = request.json
    if data and 'landmarks' in data:
        tracker.save_landmarks_to_file(data['landmarks'])
        return jsonify({'success': True, 'message': 'Данные сохранены'})
    return jsonify({'success': False, 'message': 'Неверный формат'}), 400


@app.route('/api/gesture')
def get_gesture():
    """API для получения текущего жеста"""
    global current_gesture
    return jsonify({'gesture': current_gesture})


@app.route('/api/stats')
def get_stats():
    """Статистика по сохранённым данным"""
    stats_file = 'data/hand_landmarks.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({
            'total_records': len(data),
            'file_size': os.path.getsize(stats_file)
        })
    return jsonify({'total_records': 0, 'file_size': 0})


if __name__ == '__main__':
    # Создаём необходимые директории
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    print("🚀 Сервер запущен: http://localhost:5000")
    print("📹 Откройте браузер и разрешите доступ к камере")
    print("💾 Точки рук будут сохраняться в data/hand_landmarks.json")

    app.run(debug=True, host='0.0.0.0', port=5000)