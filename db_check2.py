import psycopg2
from psycopg2.extras import DictCursor
from config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_DB)
cur = conn.cursor(cursor_factory=DictCursor)
cur.execute("SELECT id, image_b64, detections FROM asset_images WHERE asset_id = '56d3bbff-80b5-4c8d-975b-b3d3907fd214' ORDER BY id ASC")
for row in cur.fetchall():
    print('Image ID:', row['id'], 'image_filename (image_b64 field):', row['image_b64'])
    print('Detections:', row['detections'][:500] if row['detections'] else 'None')
conn.close()
