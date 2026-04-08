import sqlite3
import sys
import os
from werkzeug.security import generate_password_hash

DB_PATH = 'database.db'

def get_db():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' not found. Please run init_db.py first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def add_user(username, password, role):
    hashed_pw = generate_password_hash(password)
    conn = get_db()
    try:
        conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, hashed_pw, role))
        conn.commit()
        print(f"✅ User '{username}' added successfully as '{role}'.")
    except sqlite3.IntegrityError:
        print(f"❌ Error: Username '{username}' already exists.")
    finally:
        conn.close()

def list_users():
    conn = get_db()
    users = conn.execute('SELECT username, role FROM users').fetchall()
    conn.close()
    print("\n--- Current Application Users ---")
    for u in users:
        print(f"👤 {u['username']:<15} | Role: {u['role']}")
    print("-" * 35)

def reset_password(username, new_password):
    hashed_pw = generate_password_hash(new_password)
    conn = get_db()
    cursor = conn.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_pw, username))
    conn.commit()
    if cursor.rowcount > 0:
        print(f"✅ Password for '{username}' has been reset.")
    else:
        print(f"❌ Error: User '{username}' not found.")
    conn.close()

def delete_user(username):
    conn = get_db()
    cursor = conn.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    if cursor.rowcount > 0:
        print(f"✅ User '{username}' has been deleted.")
    else:
        print(f"❌ Error: User '{username}' not found.")
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nAsakta Vision AI - User Management Tool")
        print("Usage:")
        print("  python manage_users.py list")
        print("  python manage_users.py add [user] [pass] [admin/user]")
        print("  python manage_users.py reset [user] [new_pass]")
        print("  python manage_users.py delete [user]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    if cmd == "list":
        list_users()
    elif cmd == "add":
        if len(sys.argv) < 5:
            print("Error: Missing arguments. Use: add [username] [password] [role]")
        else:
            add_user(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "reset":
        if len(sys.argv) < 4:
            print("Error: Missing arguments. Use: reset [username] [new_password]")
        else:
            reset_password(sys.argv[2], sys.argv[3])
    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Error: Missing arguments. Use: delete [username]")
        else:
            delete_user(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
