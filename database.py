import sqlite3
import hashlib
import os
from datetime import datetime
from contextlib import contextmanager

# Путь к базе данных
DB_PATH = os.path.join(os.path.dirname(__file__), 'instance', 'users.db')


@contextmanager
def get_db():
    """Контекстный менеджер для работы с БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Инициализация базы данных - создание таблиц"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()

        # Таблица пользователей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        # Таблица для отслеживания сессий (опционально)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')

        # Таблица для логов активности (опционально)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')

        print("✅ Database initialized successfully")


def hash_password(password: str) -> str:
    """Хеширование пароля с использованием SHA-256 и соли"""
    salt = "hand_tracking_system_salt_2024"  # В реальном проекте используйте os.urandom
    return hashlib.sha256((password + salt).encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Проверка пароля"""
    return hash_password(password) == password_hash


def create_user(username: str, email: str, password: str) -> tuple:
    """
    Создание нового пользователя
    Возвращает: (success, message, user_id)
    """
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters", None

    if not email or '@' not in email:
        return False, "Invalid email address", None

    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters", None

    password_hash = hash_password(password)

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            user_id = cursor.lastrowid
            return True, "User created successfully", user_id
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return False, "Username already exists", None
        elif 'email' in str(e):
            return False, "Email already registered", None
        return False, "Database error", None


def authenticate_user(identifier: str, password: str) -> tuple:
    """
    Аутентификация пользователя по username или email
    Возвращает: (success, message, user_data)
    """
    password_hash = hash_password(password)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, email, created_at, is_active 
            FROM users 
            WHERE (username = ? OR email = ?) AND password_hash = ? AND is_active = 1
        ''', (identifier, identifier, password_hash))

        user = cursor.fetchone()

        if user:
            # Обновляем время последнего входа
            cursor.execute(
                'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
                (user['id'],)
            )
            return True, "Login successful", dict(user)
        else:
            return False, "Неверное имя пользователя/email или пароль", None


def get_user_by_id(user_id: int):
    """Получение пользователя по ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, username, email, created_at, last_login FROM users WHERE id = ?',
            (user_id,)
        )
        user = cursor.fetchone()
        return dict(user) if user else None


def log_activity(user_id: int, action: str, ip_address: str = None):
    """Логирование активности пользователя"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO user_activity (user_id, action, ip_address) VALUES (?, ?, ?)',
            (user_id, action, ip_address)
        )


def create_session(user_id: int) -> str:
    """Создание сессии для пользователя"""
    import uuid
    session_token = str(uuid.uuid4())

    with get_db() as conn:
        cursor = conn.cursor()
        # Удаляем старые сессии
        cursor.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))
        # Создаём новую
        cursor.execute(
            'INSERT INTO user_sessions (user_id, session_token) VALUES (?, ?)',
            (user_id, session_token)
        )

    return session_token


def validate_session(session_token: str) -> int:
    """Проверка валидности сессии, возвращает user_id или None"""
    if not session_token:
        return None

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT user_id FROM user_sessions WHERE session_token = ?',
            (session_token,)
        )
        result = cursor.fetchone()
        return result['user_id'] if result else None


def delete_session(session_token: str):
    """Удаление сессии (выход)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (session_token,))


def user_exists(username: str, email: str) -> tuple:
    """Проверка существования пользователя"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT username, email FROM users WHERE username = ? OR email = ?',
            (username, email)
        )
        existing = cursor.fetchone()
        if existing:
            if existing['username'] == username:
                return True, "username"
            return True, "email"
        return False, None