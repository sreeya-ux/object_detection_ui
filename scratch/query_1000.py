import psycopg2
from config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

def query_db():
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        dbname=PG_DB
    )
    cur = conn.cursor()
    
    print("--- Searching for any pole_id containing 1000 ---")
    cur.execute("SELECT id, pole_id FROM assets WHERE pole_id LIKE %s", ('%1000%',))
    print(cur.fetchall())
    
    print("\n--- Searching for any pole_id starting with RDSS ---")
    cur.execute("SELECT id, pole_id FROM assets WHERE pole_id LIKE %s", ('RDSS%',))
    for row in cur.fetchall()[:10]:
        print(row)
        
    conn.close()

if __name__ == "__main__":
    query_db()
