import sqlite3
DB_PATH = 'database.db'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('SELECT * FROM users')
rows = cursor.fetchall()
print("USER DATA DUMP")
for row in rows:
    print(f"ID: {row[0]}")
    print(f"Username: '{row[1]}' (len={len(row[1])})")
    print(f"Password: '{row[2]}' (len={len(row[2])})")
    print(f"Role: '{row[3]}' (len={len(row[3])})")
    print("-" * 20)
conn.close()
