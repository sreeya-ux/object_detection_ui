import sqlite3
import json

def audit():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    
    # 1. Total Counts
    assets = conn.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
    images = conn.execute('SELECT COUNT(*) FROM asset_images').fetchone()[0]
    print(f"Total Assets: {assets}")
    print(f"Total Images: {images}")
    
    # 2. Check for "Orphaned" Assets (Assets with no images)
    orphans = conn.execute('''
        SELECT a.id, a.worker_name, a.timestamp 
        FROM assets a
        LEFT JOIN asset_images i ON a.id = i.asset_id
        WHERE i.id IS NULL
    ''').fetchall()
    print(f"\nOrphaned Assets (No Images): {len(orphans)}")
    for o in orphans[:5]:
        print(f" - ID: {o['id']}, Worker: {o['worker_name']}, Date: {o['timestamp']}")
        
    # 3. Sample first image content info
    sample = conn.execute('SELECT image_b64 FROM asset_images LIMIT 1').fetchone()
    if sample:
        b64 = sample['image_b64']
        print(f"\nSample Image B64 Start: {b64[:50]}...")
        print(f"Length: {len(b64)}")
    else:
        print("\nNO IMAGES FOUND IN asset_images TABLE!")
        
    conn.close()

if __name__ == "__main__":
    audit()
