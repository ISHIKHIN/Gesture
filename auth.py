from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
from database import (
    authenticate_user, create_user, create_session, validate_session,
    delete_session, get_user_by_id, log_activity, user_exists, init_db
)

# Создаём Blueprint для маршрутов аутентификации
auth_bp = Blueprint('auth', __name__)

# Инициализируем БД при первом импорте
init_db()


def login_required(f):
    """Декоратор для защиты маршрутов, требующих авторизации"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = session.get('session_token')
        user_id = validate_session(session_token)

        if not user_id:
            return redirect(url_for('auth.login_page'))

        # Добавляем информацию о пользователе в запрос
        request.user_id = user_id
        request.user = get_user_by_id(user_id)

        return f(*args, **kwargs)

    return decorated_function


@auth_bp.route('/login')
def login_page():
    """Страница входа"""
    # Если уже авторизован, перенаправляем на главную
    session_token = session.get('session_token')
    if validate_session(session_token):
        return redirect(url_for('index'))
    return render_template('login.html')


@auth_bp.route('/register')
def register_page():
    """Страница регистрации"""
    session_token = session.get('session_token')
    if validate_session(session_token):
        return redirect(url_for('index'))
    return render_template('register.html')


@auth_bp.route('/api/auth/register', methods=['POST'])
def api_register():
    """API регистрации нового пользователя"""
    data = request.get_json()

    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')

    # Валидация
    if not username or len(username) < 3:
        return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400

    if not email or '@' not in email:
        return jsonify({'success': False, 'message': 'Please enter a valid email address'}), 400

    if not password or len(password) < 4:
        return jsonify({'success': False, 'message': 'Password must be at least 4 characters'}), 400

    if password != confirm_password:
        return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

    # Проверка на существование
    exists, field = user_exists(username, email)
    if exists:
        return jsonify({'success': False, 'message': f'{field.capitalize()} already exists'}), 400

    # Создание пользователя
    success, message, user_id = create_user(username, email, password)

    if success:
        return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})
    else:
        return jsonify({'success': False, 'message': message}), 400


@auth_bp.route('/api/auth/login', methods=['POST'])
def api_login():
    """API входа пользователя"""
    data = request.get_json()

    identifier = data.get('identifier', '').strip()
    password = data.get('password', '')
    remember = data.get('remember', False)

    if not identifier or not password:
        return jsonify({'success': False, 'message': 'Please fill in all fields'}), 400

    # Аутентификация
    success, message, user = authenticate_user(identifier, password)

    if success:
        # Создаём сессию
        session_token = create_session(user['id'])
        session['session_token'] = session_token

        # Логируем активность
        log_activity(user['id'], 'login', request.remote_addr)

        return jsonify({
            'success': True,
            'message': 'Login successful!',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        })
    else:
        return jsonify({'success': False, 'message': message}), 401


@auth_bp.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """API выхода пользователя"""
    session_token = session.get('session_token')
    if session_token:
        user_id = validate_session(session_token)
        if user_id:
            log_activity(user_id, 'logout', request.remote_addr)
        delete_session(session_token)
        session.pop('session_token', None)

    return jsonify({'success': True, 'message': 'Logged out successfully'})


@auth_bp.route('/api/auth/check', methods=['GET'])
def api_check_auth():
    """Проверка статуса авторизации"""
    session_token = session.get('session_token')
    user_id = validate_session(session_token)

    if user_id:
        user = get_user_by_id(user_id)
        return jsonify({
            'authenticated': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        })
    else:
        return jsonify({'authenticated': False})