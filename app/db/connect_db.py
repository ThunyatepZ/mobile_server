# database.py
import psycopg2
import os

def get_connection():
    database_url = os.getenv("SUPERBASE_CONNECTION")
    if not database_url:
        raise ValueError("SUPERBASE_CONNECTION is not set in .env file!")
    return psycopg2.connect(
        database_url,
        sslmode="require"
    )