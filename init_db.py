from werkzeug.security import generate_password_hash
import sqlite3
import os

DB_PATH = 'database.db'

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    ''')

    # Create Assets table (Workflow Header)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        worker_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', -- pending, approved, rejected
        timestamp TEXT NOT NULL,
        asset_class TEXT,                       -- From Rule Engine
        voltage TEXT,
        reason TEXT
    )
    ''')

    # Create Asset Images table (Inference Results)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS asset_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT NOT NULL,
        image_b64 TEXT NOT NULL,
        detections TEXT NOT NULL,               -- JSON string of boxes
        pole_angle FLOAT DEFAULT 0.0,           -- For leaning pole detection
        FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
    )
    ''')

    # Create Activity Logs table (Requirement 1.4)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TEXT NOT NULL
    )
    ''')

    # Add initial users with hashed passwords
    users = [
        ('admin', generate_password_hash('admin123'), 'admin'),
        ('Worker-Alpha', generate_password_hash('worker'), 'user'),
    ]
    cursor.executemany('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', users)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
