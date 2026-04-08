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

    # Create Tasks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        worker_name TEXT NOT NULL,
        image_b64 TEXT NOT NULL,
        detections TEXT NOT NULL, -- JSON string
        status TEXT NOT NULL DEFAULT 'pending',
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
