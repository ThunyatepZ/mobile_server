from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from typing import List
import shutil
import os
import uuid

from app.db.superbase import get_db_connection
from app.api.endpoint.auth import oauth2_scheme, decoder_token
from app.service.quiz_service import (
    extract_text_from_pdf, 
    generate_quiz_from_text, 
    save_quiz_to_db
)

router = APIRouter()

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# Temporary upload folder
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/generate")
async def generate_quiz(
    file: UploadFile = File(...), 
    token: str = Depends(oauth2_scheme), 
    conn=Depends(get_db)
):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # Get user id
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s;", (token_data.email,))
    user_row = cursor.fetchone()
    if not user_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_id = user_row[0]
    cursor.close()

    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract and Generate
        text = ""
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r") as f:
                text = f.read()
        
        # Generate Quiz with AI
        quiz_json = generate_quiz_from_text(text)
        
        # Save to DB
        quiz_id = save_quiz_to_db(conn, user_id, quiz_json)
        
        return {"status": "success", "quiz_id": str(quiz_id), "title": quiz_json['title']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.get("/")
def get_all_quizzes(conn=Depends(get_db)):
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT q.id, q.title, q.description, u.username, q.created_at
            FROM quizzes q
            JOIN users u ON q.creator_id = u.id
            WHERE q.is_public = true
            ORDER BY q.created_at DESC;
            """
        )
        rows = cursor.fetchall()
        quizzes = []
        for row in rows:
            quizzes.append({
                "id": str(row[0]),
                "title": row[1],
                "description": row[2],
                "author": row[3],
                "created_at": str(row[4])
            })
        return {"status": "success", "quizzes": quizzes}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()

@router.get("/{quiz_id}")
def get_quiz_detail(quiz_id: str, conn=Depends(get_db)):
    cursor = conn.cursor()
    try:
        # Get Quiz
        cursor.execute("SELECT title, description FROM quizzes WHERE id = %s;", (quiz_id,))
        quiz_row = cursor.fetchone()
        if not quiz_row:
            raise HTTPException(status_code=404, detail="Quiz not found")

        # Get Questions
        cursor.execute(
            "SELECT id, question_text, options, correct_answer, explanation FROM questions WHERE quiz_id = %s;", 
            (quiz_id,)
        )
        q_rows = cursor.fetchall()
        questions = []
        for r in q_rows:
            questions.append({
                "id": str(r[0]),
                "question_text": r[1],
                "options": r[2], # Already JSON in DB (JSONB)
                "correct_answer": r[3],
                "explanation": r[4]
            })
        
        return {
            "status": "success", 
            "title": quiz_row[0], 
            "description": quiz_row[1], 
            "questions": questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
