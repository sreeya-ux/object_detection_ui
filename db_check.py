import sqlite3
import psycopg2
from psycopg2.extras import DictCursor

print('=== Checking SQLite ===')
try:
    conn = sqlite3.connect('/home/ubuntu/object_detection_ui/database.db')
    r = conn.execute('SELECT id, timestamp, status, asset_class, voltage, pole_id FROM assets ORDER BY timestamp DESC LIMIT 3').fetchall()
    print('SQLite assets:')
    for row in r:
        print(row)
    r2 = conn.execute('SELECT id, asset_id, pole_angle FROM asset_images ORDER BY id DESC LIMIT 5').fetchall()
    print('SQLite images:')
    for row in r2:
        print(row)
    conn.close()
except Exception as e:
    print('SQLite error:', e)

print('\n=== Checking PostgreSQL ===')
try:
    from config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB
    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_DB)
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute('SELECT id, timestamp, status, asset_class, voltage, pole_id FROM assets ORDER BY timestamp DESC LIMIT 3')
    print('Postgres assets:')
    for row in cur.fetchall():
        print(dict(row))
    cur.execute('SELECT id, asset_id, pole_angle FROM asset_images ORDER BY id DESC LIMIT 5')
    print('Postgres images:')
    for row in cur.fetchall():
        print(dict(row))
    conn.close()
except Exception as e:
    print('Postgres error:', e)

