from fastapi import FastAPI
from dotenv import load_dotenv
import os
import sys


from app.db.superbase import get_db_connection
from app.api.router import api_router

load_dotenv()

app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("connecting to database...")
    conn = get_db_connection()
    if conn:
        print("Database connected")
        conn.close()
    else:
        print("Database connection failed.")

app.include_router(api_router, prefix="/api/v1")
