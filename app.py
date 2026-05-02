from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
import json
import os
import time
from hand_tracker import HandTracker
from threading import Lock
import platform
import subprocess

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'

tracker = HandTracker(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

current_landmarks = []
current_gesture = "Waiting"
current_meme = "none"
landmarks_lock = Lock()

fps_counter = 0
last_fps_time = time.time()
actual_fps = 30

FRAME_SKIP = 0
frame_counter = 0

mirror_enabled = True
current_camera_id = 0
camera_initialized = False
generator_thread = None
generator_lock = Lock()

# Обновленный словарь соответствия жестов и мемов
# Обновленный словарь соответствия жестов и мемов
GESTURE_TO_MEME = {
    # Существующие жесты
    "Pointing Up": 'pic_finger_up.jpg',
    "Thumbs Up": "pic_thumb_up.jpg",
    "Peace": "pic_peace.jpg",

    # Новые жесты
    "Little": "fingers_little.jpeg",
    "Two Palms Parallel": "2_hands_infrontofeachother.jpg",
    "I Don't Know": "i_don`t_know.jpg",
    "Open Palm": "open_palm.jpg",
    "Cherry": "cherry_pic.jpg"
}


def get_available_cameras():
    """Определяет доступные камеры на системе"""
    available_cameras = []

    # Пробуем разные бэкенды для разных ОС
    backends = []
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif platform.system() == "Darwin":  # macOS
        backends = [cv2.CAP_ANY]
    else:  # Linux
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    # Проверяем индексы камер от 0 до 9
    for i in range(10):
        for backend in backends:
            try:
                # Пробуем открыть камеру с разными бэкендами
                if backend != cv2.CAP_ANY:
                    cap = cv2.VideoCapture(i, backend)
                else:
                    cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    # Успешно открыли камеру
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Пытаемся получить название камеры
                    camera_name = get_camera_name(i)

                    # Проверяем, не добавлена ли уже такая камера
                    if not any(cam['id'] == i for cam in available_cameras):
                        available_cameras.append({
                            'id': i,
                            'name': camera_name,
                            'resolution': f"{width}x{height}",
                            'backend': str(backend)
                        })

                    cap.release()
                    break  # Нашли работающий бэкенд для этой камеры
            except Exception as e:
                continue

    # Если камер не найдено, пробуем специфичные для DroidCam пути
    if not available_cameras:
        # DroidCam обычно использует индекс 1 или 2
        for i in [1, 2, 3, 4]:
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available_cameras.append({
                        'id': i,
                        'name': f"DroidCam (index {i})",
                        'resolution': f"{width}x{height}",
                        'backend': 'auto'
                    })
                    cap.release()
            except:
                pass

    return available_cameras


def get_camera_name(camera_id):
    """Пытается получить понятное имя камеры"""
    if platform.system() == "Windows":
        try:
            # На Windows можно использовать PowerShell для получения имени камеры
            result = subprocess.run(
                ['powershell', '-Command',
                 f'Get-PnpDevice -Class Image | Where-Object {{ $_.FriendlyName -like "*camera*" }} | Select-Object -First 1 -ExpandProperty FriendlyName'],
                capture_output=True, text=True, timeout=2
            )
            if result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'],
                                    capture_output=True, text=True, timeout=2)
            if "FaceTime" in result.stdout:
                return "FaceTime HD Camera"
        except:
            pass

    # Стандартные имена для разных типов камер
    if camera_id == 0:
        return "Built-in Camera"
    elif camera_id == 1:
        return "External Camera / DroidCam"
    elif camera_id == 2:
        return "Secondary Camera"
    else:
        return f"Camera {camera_id}"


def init_camera(camera_id):
    """Инициализирует камеру с указанным ID и возвращает объект VideoCapture"""
    try:
        # Пробуем разные бэкенды для лучшей совместимости
        if platform.system() == "Windows":
            # Сначала пробуем DirectShow (лучше для DroidCam)
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Пробуем MSMF
                cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {camera_id}")
            return None

        # Настройки камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Проверяем реальные настройки
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps_cam = cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera {camera_id} initialized: {actual_width}x{actual_height} @ {actual_fps_cam}fps")
        return cap

    except Exception as e:
        print(f"Error initializing camera {camera_id}: {e}")
        return None


def generate_frames():
    """Генератор кадров с переключением камеры"""
    global current_landmarks, current_gesture, current_meme, fps_counter, last_fps_time, actual_fps, frame_counter, current_camera_id

    cap = None

    while True:
        try:
            # Если камера не инициализирована или закрыта, создаем новую
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()

                cap = init_camera(current_camera_id)

                if cap is None:
                    # Показываем черный экран с сообщением
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, f"Camera {current_camera_id} not available", (150, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(black_frame, "Check camera connection or select another camera", (100, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.1)
                    continue

            # Читаем кадр
            success, frame = cap.read()

            if not success:
                print(f"ERROR: Failed to read frame from camera {current_camera_id}")
                cap.release()
                cap = None
                time.sleep(0.1)
                continue

            if mirror_enabled:
                frame = cv2.flip(frame, 1)

            # Пропуск кадров для производительности
            frame_counter += 1
            if frame_counter <= FRAME_SKIP:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue
            else:
                frame_counter = 0

            # Обработка кадра для обнаружения рук
            annotated_frame, landmarks = tracker.process_frame(frame)

            with landmarks_lock:
                current_landmarks = landmarks
                if landmarks:
                    current_gesture = tracker.recognize_gesture(landmarks)
                    if current_gesture in GESTURE_TO_MEME:
                        current_meme = GESTURE_TO_MEME[current_gesture]
                    else:
                        current_meme = "none"
                else:
                    current_gesture = "No hand"
                    current_meme = "none"

            # Отображаем информацию на кадре
            cv2.putText(annotated_frame, f"Gesture: {current_gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2)
            cv2.putText(annotated_frame, f"Hands: {len(landmarks)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136, 136, 136), 1)
            cv2.putText(annotated_frame, f"FPS: {actual_fps}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Camera ID: {current_camera_id}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            # Кодируем кадр в JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            # Подсчет FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                actual_fps = fps_counter
                fps_counter = 0
                last_fps_time = current_time

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.1)
            continue


# Глобальная переменная для генератора
frame_generator = None


def get_frame_generator():
    """Возвращает единственный экземпляр генератора кадров"""
    global frame_generator
    if frame_generator is None:
        frame_generator = generate_frames()
    return frame_generator


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/video_feed')
def video_feed():
    return Response(get_frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/landmarks')
def get_landmarks():
    with landmarks_lock:
        return jsonify({
            'hands': current_landmarks,
            'gesture': current_gesture,
            'meme': current_meme,
            'fps': actual_fps,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })


@app.route('/api/save_landmarks', methods=['POST'])
def save_landmarks():
    with landmarks_lock:
        if current_landmarks:
            index = tracker.save_landmarks_to_file(current_landmarks)
            return jsonify({
                'success': True,
                'message': f'Data saved (record #{index})',
                'record_id': index
            })
    return jsonify({'success': False, 'message': 'No data to save'}), 400


@app.route('/api/save_custom', methods=['POST'])
def save_custom_landmarks():
    data = request.json
    if data and 'landmarks' in data:
        tracker.save_landmarks_to_file(data['landmarks'])
        return jsonify({'success': True, 'message': 'Data saved'})
    return jsonify({'success': False, 'message': 'Invalid format'}), 400


@app.route('/api/gesture')
def get_gesture():
    with landmarks_lock:
        return jsonify({'gesture': current_gesture})


@app.route('/api/meme')
def get_meme():
    with landmarks_lock:
        return jsonify({'meme': current_meme})


@app.route('/api/stats')
def get_stats():
    stats_file = 'data/hand_landmarks.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({
            'total_records': len(data),
            'file_size': os.path.getsize(stats_file),
            'current_fps': actual_fps
        })
    return jsonify({'total_records': 0, 'file_size': 0, 'current_fps': actual_fps})


@app.route('/api/settings', methods=['POST'])
def update_settings():
    global FRAME_SKIP, mirror_enabled
    data = request.json

    if 'frame_skip' in data:
        FRAME_SKIP = data['frame_skip']
    if 'mirror' in data:
        mirror_enabled = data['mirror']

    return jsonify({
        'frame_skip': FRAME_SKIP,
        'mirror': mirror_enabled
    })


@app.route('/api/cameras')
def get_cameras():
    """Возвращает список доступных камер"""
    cameras = get_available_cameras()
    return jsonify({
        'cameras': cameras,
        'current_camera': current_camera_id
    })


@app.route('/api/switch_camera', methods=['POST'])
def switch_camera():
    """Переключает на указанную камеру"""
    global current_camera_id, frame_generator

    data = request.json
    camera_id = data.get('camera_id', 0)

    if camera_id == current_camera_id:
        return jsonify({'success': True, 'message': 'Camera already selected'})

    # Проверяем, доступна ли камера
    test_cap = init_camera(camera_id)
    if test_cap is None:
        return jsonify({
            'success': False,
            'message': f'Cannot open camera {camera_id}'
        }), 400

    test_cap.release()

    # Переключаем камеру
    current_camera_id = camera_id

    # Сбрасываем генератор, чтобы он создал новый с новой камерой
    frame_generator = None

    return jsonify({
        'success': True,
        'message': f'Switched to camera {camera_id}',
        'camera_id': camera_id
    })


@app.route('/api/camera_info')
def camera_info():
    """Возвращает информацию о текущей камере"""
    return jsonify({
        'camera_id': current_camera_id,
        'available_cameras': len(get_available_cameras())
    })


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    print("=" * 50)
    print("Starting Hand Gesture Recognition System")
    print("=" * 50)

    # Проверяем доступные камеры
    cameras = get_available_cameras()
    print(f"\nAvailable cameras: {len(cameras)}")
    for cam in cameras:
        print(f"  - Camera {cam['id']}: {cam['name']} ({cam['resolution']})")

    if cameras:
        print(f"\nUsing default camera: {cameras[0]['name']}")
        current_camera_id = cameras[0]['id']
    else:
        print("\nWARNING: No cameras found! Please check your camera connection.")
        print("For DroidCam, make sure:")
        print("  1. DroidCam app is running on your phone")
        print("  2. USB debugging is enabled (for USB connection)")
        print("  3. Both devices are on same WiFi (for WiFi connection)")

    print("\n" + "=" * 50)
    print("Server started at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)