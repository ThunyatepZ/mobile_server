from fastapi import FastAPI
from app.api.router import api_router
from dotenv import load_dotenv
import os

load_dotenv()
from app.db.connect_db import get_connection

app = FastAPI()

print(os.getenv("SUPERBASE_CONNECTION"))

@app.on_event("startup")
def startup_event():
    print("Application startup...")
    try:
        conn = get_connection()
        print("Database connected!")
        conn.close()
    except Exception as e:
        print(f"WARNING: Database connection failed: {e}")
        print("App will continue running without database connection.")

app.include_router(api_router,prefix="/api/v1")
