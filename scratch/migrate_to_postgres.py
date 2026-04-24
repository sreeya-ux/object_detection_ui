
import sqlite3
import psycopg2
from psycopg2.extras import execute_values
import os

# --- CONFIGURATION (Match your config.py) ---
SQLITE_DB = 'database.db'
PG_CONFIG = {
    'host': 'localhost',
    'port': '5434',
    'database': 'infrastructure_db',
    'user': 'postgres',
    'password': 'password'
}

def migrate():
    print("--- Starting Migration: SQLite -> PostgreSQL ---")
    
    if not os.path.exists(SQLITE_DB):
        print("Error: SQLite database not found!")
        return

    # 1. Connect
    sl_conn = sqlite3.connect(SQLITE_DB)
    sl_conn.row_factory = sqlite3.Row
    sl_cur = sl_conn.cursor()

    pg_conn = psycopg2.connect(**PG_CONFIG)
    pg_cur = pg_conn.cursor()

    # 2. Create Tables in Postgres
    print("Creating tables in PostgreSQL...")
    pg_cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            worker_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            timestamp TEXT NOT NULL,
            asset_class TEXT,
            voltage TEXT,
            reason TEXT
        );
        CREATE TABLE IF NOT EXISTS asset_images (
            id SERIAL PRIMARY KEY,
            asset_id TEXT NOT NULL,
            image_b64 TEXT NOT NULL,
            detections TEXT NOT NULL,
            pole_angle FLOAT DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS activity_logs (
            id SERIAL PRIMARY KEY,
            user_name TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TEXT NOT NULL
        );
    ''')
    pg_conn.commit()

    # 3. Migrate Tables
    tables = ['users', 'assets', 'asset_images', 'activity_logs']
    
    for table in tables:
        print(f"Migrating table: {table}...")
        sl_cur.execute(f"SELECT * FROM {table}")
        rows = sl_cur.fetchall()
        
        if not rows:
            print(f"  (Table {table} is empty, skipping)")
            continue

        # Prepare columns and placeholders
        colnames = rows[0].keys()
        cols_str = ",".join(colnames)
        placeholders = ",".join(["%s"] * len(colnames))
        
        data = [tuple(row) for row in rows]
        
        insert_query = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        pg_cur.executemany(insert_query, data)
        print(f"  Successfully moved {len(rows)} rows.")

    pg_conn.commit()
    sl_conn.close()
    pg_conn.close()
    print("--- Migration Complete! ---")

if __name__ == "__main__":
    migrate()
