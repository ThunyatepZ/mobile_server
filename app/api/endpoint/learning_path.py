from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
import json

from app.api.endpoint.auth import get_current_user
from app.db.superbase import get_db_connection

router = APIRouter()

class PathProgressUpdate(BaseModel):
    path_id: str
    completed_lessons: List[str]

@router.get("/progress/{path_id}")
async def get_path_progress(path_id: str, current_user: str = Depends(get_current_user)):
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT completed_lessons FROM learning_path_progress WHERE user_id = %s AND path_id = %s",
            (current_user, path_id)
        )
        row = cursor.fetchone()
        
        if row:
            completed_lessons = row[0] if row[0] is not None else []
            return {
                "success": True,
                "is_enrolled": True,
                "completed_lessons": completed_lessons,
            }
        return {"success": True, "is_enrolled": False, "completed_lessons": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

@router.post("/progress")
async def update_path_progress(progress: PathProgressUpdate, current_user: str = Depends(get_current_user)):
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
        
    cursor = None
    try:
        cursor = connection.cursor()
        completed_lessons_json = json.dumps(progress.completed_lessons)
        
        cursor.execute(
            """
            INSERT INTO learning_path_progress (user_id, path_id, completed_lessons)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (user_id, path_id) 
            DO UPDATE SET completed_lessons = EXCLUDED.completed_lessons
            """,
            (current_user, progress.path_id, completed_lessons_json)
        )
        connection.commit()
        return {"success": True, "message": "Progress updated successfully"}
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()
