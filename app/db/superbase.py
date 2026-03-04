import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

def get_db_connection():
    # Fetch variables
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    # Connect to the database
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        return connection
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None

if __name__ == "__main__":
    # Test connection if run directly
    conn = get_db_connection()
    if conn:
        print("✅ Connection successful!")
        cursor = conn.cursor()
        cursor.execute("SELECT NOW();")
        print("🕒 Current Time:", cursor.fetchone())
        cursor.close()
        conn.close()
        print("🔌 Connection closed.")