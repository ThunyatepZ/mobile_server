from app.db.superbase import get_db_connection

conn = get_db_connection()
if conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM courses LIMIT 3")
    print("Courses:", cursor.fetchall())
    cursor.execute("SELECT * FROM enrollments LIMIT 3")
    print("Enrollments:", cursor.fetchall())
    cursor.close()
    conn.close()
