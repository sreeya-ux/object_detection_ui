import psycopg2
from psycopg2.extras import RealDictCursor
from config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

def check_db():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        dbname=PG_DB,
        cursor_factory=RealDictCursor
    )
    with conn.cursor() as cur:
        # Check latest assets
        print("\n--- Latest Assets ---")
        cur.execute("SELECT * FROM assets ORDER BY timestamp DESC LIMIT 5")
        for row in cur.fetchall():
            print(dict(row))
            
        # Check columns of asset_images
        print("\n--- Asset Images Columns ---")
        cur.execute("SELECT * FROM asset_images LIMIT 1")
        row = cur.fetchone()
        if row:
            print(list(row.keys()))
            print({k: v for k, v in row.items() if k != 'image_b64'})
            
        # Check latest asset_images
        print("\n--- Latest Asset Images ---")
        cur.execute("SELECT id, asset_id, SUBSTRING(detections FROM 1 FOR 200) AS dets_trunc FROM asset_images ORDER BY id DESC LIMIT 5")
        for row in cur.fetchall():
            print(dict(row))

    conn.close()

if __name__ == "__main__":
    check_db()
