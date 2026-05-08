import sqlite3
from werkzeug.security import generate_password_hash

DB_PATH = 'database.db'

def patch_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Update Admin (admin -> admin 1)
    cursor.execute("UPDATE users SET username = ?, password = ? WHERE username = 'admin'", 
                   ('admin 1', generate_password_hash('admin@asakta')))
    
    # 2. Update Worker (Worker-Alpha -> user 1)
    cursor.execute("UPDATE users SET username = ?, password = ? WHERE username = 'Worker-Alpha' OR username = 'worker-alpha'", 
                   ('user 1', generate_password_hash('1233@asakta')))
    
    conn.commit()
    count = cursor.rowcount
    conn.close()
    
    print(f"Patched {count} user rows successfully.")

if __name__ == "__main__":
    patch_users()
