import sqlite3
DB_PATH = 'database.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('SELECT * FROM users')
rows = cursor.fetchall()
print("ID | Username | Password | Role")
print("-" * 30)
for row in rows:
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
conn.close()
