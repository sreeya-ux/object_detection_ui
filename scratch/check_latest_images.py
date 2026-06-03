import psycopg2
from psycopg2.extras import RealDictCursor
from config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

def check_images():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        dbname=PG_DB,
        cursor_factory=RealDictCursor
    )
    with conn.cursor() as cur:
        cur.execute("SELECT id, timestamp, asset_class, pole_id FROM assets ORDER BY timestamp DESC LIMIT 10")
        assets = cur.fetchall()
        print("\n--- Latest Assets ---")
        for asset in assets:
            print(f"Asset ID: {asset['id']}")
            print(f"  Timestamp: {asset['timestamp']}")
            print(f"  Class: {asset['asset_class']}")
            print(f"  Pole ID: {repr(asset['pole_id'])}")
            
            # Query images for this asset
            cur.execute("SELECT id, SUBSTRING(CAST(detections AS text) FROM 1 FOR 150) AS dets_txt FROM asset_images WHERE asset_id = %s", (asset['id'],))
            imgs = cur.fetchall()
            print(f"  Images ({len(imgs)}):")
            for img in imgs:
                print(f"    Image ID: {img['id']}")
                print(f"      Detections (trunc): {img['dets_txt']}...")
                
    conn.close()

if __name__ == "__main__":
    check_images()
