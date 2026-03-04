from app.db.superbase import get_db_connection

conn = get_db_connection()
if conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = cursor.fetchall()
    print("Tables:")
    for table in tables:
        print(f" - {table[0]}")
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table[0]}'")
        for c in cursor.fetchall():
            print(f"    {c[0]} ({c[1]})")
    cursor.close()
    conn.close()
