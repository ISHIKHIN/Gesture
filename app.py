from flask import Flask, render_template, Response, jsonify, request, send_from_directory, session, redirect, url_for
import cv2
import numpy as np
import json
import os
import time
from hand_tracker import HandTracker
from threading import Lock, Thread
import platform
import subprocess

from auth import auth_bp, login_required, validate_session
from database import init_db, get_user_by_id

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this-in-production-2024'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

app.register_blueprint(auth_bp)

init_db()

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

mirror_enabled = True
current_camera_id = 0

latest_frame = None
frame_lock = Lock()
camera_running = True
camera_thread = None

GESTURE_TO_MEME = {
    "Pointing Up": 'pic_finger_up.jpg',
    "Thumbs Up": "pic_thumb_up.jpg",
    "Peace": "pic_peace.jpg",
    "Little": "2_fingers_little.jpg",
    "Fist": "fist.jpg",
    "Face Palm": "face_palm.jpg",
    "Two Palms Parallel": "2_hands_infrontofeachother.jpg",
    "I Don't Know": "i_don`t_know.jpg",
    "Open Palm": "open_palm.jpg",
    "Cherry": "cherry_pic.jpg"
}


def get_available_cameras():
    available_cameras = []
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                camera_name = get_camera_name(i)
                available_cameras.append({
                    'id': i,
                    'name': camera_name,
                    'resolution': f"{width}x{height}"
                })
                cap.release()
        except:
            pass
    if not available_cameras:
        available_cameras.append({'id': 0, 'name': 'Default Camera', 'resolution': '640x480'})
    return available_cameras


def get_camera_name(camera_id):
    if platform.system() == "Windows":
        try:
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
    if camera_id == 0:
        return "Built-in Camera"
    elif camera_id == 1:
        return "External Camera"
    else:
        return f"Camera {camera_id}"


def init_camera(camera_id):
    try:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    except Exception as e:
        print(f"Error initializing camera {camera_id}: {e}")
        return None


def camera_processing_thread():
    global latest_frame, current_landmarks, current_gesture, current_meme, actual_fps, fps_counter, last_fps_time, camera_running, current_camera_id
    cap = None
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 2
    print("Camera thread started")
    while camera_running:
        try:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = init_camera(current_camera_id)
                if cap is None:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Camera {current_camera_id} not available", (150, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2)
                    cv2.putText(error_frame, "Please select another camera", (200, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 1)
                    cv2.putText(error_frame, f"FPS: {actual_fps}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 1)
                    ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with frame_lock:
                        latest_frame = buffer.tobytes()
                    time.sleep(0.5)
                    continue
            success, frame = cap.read()
            if not success:
                print(f"Failed to read frame from camera {current_camera_id}")
                time.sleep(0.1)
                continue
            if mirror_enabled:
                frame = cv2.flip(frame, 1)
            frame_counter += 1
            if frame_counter >= PROCESS_EVERY_N_FRAMES:
                frame_counter = 0
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
                        current_gesture = "Нет рук"
                        current_meme = "none"
                output_frame = annotated_frame
            else:
                output_frame = frame.copy()
                with landmarks_lock:
                    if current_landmarks and len(current_landmarks) > 0:
                        h, w, _ = output_frame.shape
                        for hand_data in current_landmarks:
                            points = hand_data['landmarks']
                            connections = [
                                (0, 1), (1, 2), (2, 3), (3, 4),
                                (0, 5), (5, 6), (6, 7), (7, 8),
                                (5, 9), (9, 10), (10, 11), (11, 12),
                                (9, 13), (13, 14), (14, 15), (15, 16),
                                (13, 17), (17, 18), (18, 19), (19, 20),
                                (0, 17), (5, 9), (9, 13), (13, 17)
                            ]
                            for connection in connections:
                                start_idx, end_idx = connection
                                if start_idx < len(points) and end_idx < len(points):
                                    start_point = (points[start_idx]['x'], points[start_idx]['y'])
                                    end_point = (points[end_idx]['x'], points[end_idx]['y'])
                                    cv2.line(output_frame, start_point, end_point, (255, 200, 0), 2)
                            for point in points:
                                cv2.circle(output_frame, (point['x'], point['y']), 4, (255, 100, 0), -1)
                                cv2.circle(output_frame, (point['x'], point['y']), 4, (255, 255, 255), 1)
            with landmarks_lock:
                gesture_text = current_gesture
                hands_count = len(current_landmarks)
            title_blue_bgr = (246, 130, 59)
            cv2.putText(output_frame, f"Gesture: {gesture_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_blue_bgr, 2)
            cv2.putText(output_frame, f"Hands: {hands_count}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_blue_bgr, 1)
            cv2.putText(output_frame, f"FPS: {actual_fps}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_blue_bgr, 1)
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with frame_lock:
                latest_frame = buffer.tobytes()
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                actual_fps = fps_counter
                fps_counter = 0
                last_fps_time = current_time
        except Exception as e:
            print(f"Error in camera processing: {e}")
            time.sleep(0.1)
            continue


def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame_bytes = latest_frame
            else:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                title_blue_bgr = (246, 130, 59)
                cv2.putText(black_frame, "Waiting for camera...", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_blue_bgr, 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/landmarks')
@login_required
def get_landmarks():
    with landmarks_lock:
        return jsonify({
            'hands': current_landmarks,
            'gesture': current_gesture,
            'meme': current_meme,
            'fps': actual_fps,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })


@app.route('/api/gesture')
@login_required
def get_gesture():
    with landmarks_lock:
        return jsonify({'gesture': current_gesture})


@app.route('/api/meme')
@login_required
def get_meme():
    with landmarks_lock:
        return jsonify({'meme': current_meme})


@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    global mirror_enabled
    data = request.json
    if 'mirror' in data:
        mirror_enabled = data['mirror']
    return jsonify({'mirror': mirror_enabled})


@app.route('/api/cameras')
@login_required
def get_cameras():
    cameras = get_available_cameras()
    return jsonify({
        'cameras': cameras,
        'current_camera': current_camera_id
    })


@app.route('/api/switch_camera', methods=['POST'])
@login_required
def switch_camera():
    global current_camera_id
    data = request.json
    camera_id = data.get('camera_id', 0)
    if camera_id == current_camera_id:
        return jsonify({'success': True, 'message': 'Camera already selected'})
    test_cap = init_camera(camera_id)
    if test_cap is None:
        return jsonify({
            'success': False,
            'message': f'Cannot open camera {camera_id}'
        }), 400
    test_cap.release()
    current_camera_id = camera_id
    return jsonify({
        'success': True,
        'message': f'Switched to camera {camera_id}',
        'camera_id': camera_id
    })


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('instance', exist_ok=True)

    print("=" * 50)
    print("Starting Hand Gesture Recognition System")
    print("=" * 50)

    cameras = get_available_cameras()
    print(f"\nAvailable cameras: {len(cameras)}")
    for cam in cameras:
        print(f"  - Camera {cam['id']}: {cam['name']} ({cam['resolution']})")

    if cameras:
        print(f"\nUsing default camera: {cameras[0]['name']}")
        current_camera_id = cameras[0]['id']
    else:
        print("\nWARNING: No cameras found! Please check your camera connection.")

    camera_thread = Thread(target=camera_processing_thread, daemon=True)
    camera_thread.start()
    print("Camera thread started")

    print("\n" + "=" * 50)
    print("Server started at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)