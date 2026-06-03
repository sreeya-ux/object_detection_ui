import psycopg2
from config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

def init_postgres():
    try:
        # Connect to the default 'postgres' database first to create the target DB if it doesn't exist
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, database='postgres'
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Create the database if it doesn't exist
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{PG_DB}'")
        exists = cur.fetchone()
        if not exists:
            print(f"Creating database {PG_DB}...")
            cur.execute(f"CREATE DATABASE {PG_DB}")
        
        cur.close()
        conn.close()

        # Now connect to the actual target database
        print(f"Connecting to {PG_DB} to initialize schema...")
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, database=PG_DB
        )
        cur = conn.cursor()

        # SQL Schema
        schema = """
        -- 1. Create Users Table
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        );

        -- 2. Create Assets Table
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            worker_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            timestamp TIMESTAMP NOT NULL,
            asset_class TEXT,
            voltage TEXT,
            reason TEXT,
            pole_id TEXT
        );

        -- 3. Create Asset Images Table
        CREATE TABLE IF NOT EXISTS asset_images (
            id SERIAL PRIMARY KEY,
            asset_id TEXT NOT NULL,
            image_b64 TEXT NOT NULL,
            detections JSONB NOT NULL,
            pole_angle FLOAT DEFAULT 0.0,
            CONSTRAINT fk_asset FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );

        -- 4. Create Activity Logs Table
        CREATE TABLE IF NOT EXISTS activity_logs (
            id SERIAL PRIMARY KEY,
            user_name TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP NOT NULL
        );

        -- 5. Create Training Samples Table
        CREATE TABLE IF NOT EXISTS training_samples (
            id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            image_file TEXT,
            label_file TEXT,
            class_counts JSONB,
            approved_by TEXT,
            timestamp TIMESTAMP NOT NULL
        );
        
        -- 6. Create Training Runs Table
        CREATE TABLE IF NOT EXISTS training_runs (
            id SERIAL PRIMARY KEY,
            triggered_at TIMESTAMP,
            sample_count INTEGER,
            status TEXT DEFAULT 'queued',
            result TEXT
        );
        """
        
        cur.execute(schema)
        
        # Add default admin if not exists (admin / admin@asakta)
        # Note: Password hash is for 'admin@asakta' using Werkzeug
        from werkzeug.security import generate_password_hash
        hashed_pw = generate_password_hash('admin@asakta')
        
        cur.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            print("Adding default admin user...")
            cur.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", 
                        ('admin', hashed_pw, 'admin'))
            
        conn.commit()
        cur.close()
        conn.close()
        print("✅ PostgreSQL Database initialized successfully!")

    except Exception as e:
        print(f"❌ Error initializing PostgreSQL: {e}")

if __name__ == "__main__":
    init_postgres()
