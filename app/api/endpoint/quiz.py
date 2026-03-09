from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from typing import List
import shutil
import os
import uuid

from app.db.superbase import get_db_connection
from app.api.endpoint.auth import oauth2_scheme
from app.core.jwt_handler import decoder_token
from app.service.quiz_service import (
    extract_text_from_pdf, 
    generate_quiz_from_text, 
    save_quiz_to_db
)

class QuizSubmitRequest(BaseModel):
    quiz_id: str
    score: int
    total_questions: int

router = APIRouter()

def get_db():
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    try:
        yield connection
    finally:
        connection.close()

# Temporary upload folder
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _get_user_id_from_email(connection, email: str):
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s;", (email,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()

@router.post("/generate")
async def generate_quiz(
    files: List[UploadFile] = File(...), 
    token: str = Depends(oauth2_scheme), 
    conn=Depends(get_db)
):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = _get_user_id_from_email(conn, token_data.email)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    combined_text = ""
    temporary_file_paths: List[str] = []

    try:
        for upload in files:
            file_id = str(uuid.uuid4())
            temp_path = os.path.join(UPLOAD_DIR, f"{file_id}_{upload.filename}")
            temporary_file_paths.append(temp_path)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            
            if upload.filename.endswith(".pdf"):
                combined_text += extract_text_from_pdf(temp_path) + "\n"
            else:
                with open(temp_path, "r", encoding="utf-8") as f:
                    combined_text += f.read() + "\n"
        
        quiz_json = generate_quiz_from_text(combined_text)
        
        quiz_id = save_quiz_to_db(conn, user_id, quiz_json)
        
        return {"status": "success", "quiz_id": str(quiz_id), "title": quiz_json['title']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for temp_path in temporary_file_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

@router.get("/")
def get_all_quizzes(token: str = Depends(oauth2_scheme), conn=Depends(get_db)):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s;", (token_data.email,))
        user_row = cursor.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = user_row[0]

        cursor.execute(
            """
            SELECT q.id, q.title, q.description, u.username, q.created_at
            FROM quizzes q
            JOIN users u ON q.creator_id = u.id
            WHERE q.creator_id = %s
            ORDER BY q.created_at DESC;
            """,
            (user_id,)
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


@router.get("/my")
def get_my_quizzes(token: str = Depends(oauth2_scheme), conn=Depends(get_db)):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    cursor = conn.cursor()
    try:
        user_id = _get_user_id_from_email(conn, token_data.email)
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")

        cursor.execute(
            """
            SELECT id, title, description, created_at, is_public
            FROM quizzes
            WHERE creator_id = %s
            ORDER BY created_at DESC;
            """,
            (user_id,)
        )
        rows = cursor.fetchall()
        quizzes = []
        for row in rows:
            quizzes.append({
                "id": str(row[0]),
                "title": row[1],
                "description": row[2],
                "created_at": str(row[3]),
                "is_public": row[4]
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

@router.post("/submit")
def submit_quiz_attempt(req: QuizSubmitRequest, token: str = Depends(oauth2_scheme), conn=Depends(get_db)):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    cursor = conn.cursor()
    try:
        user_id = _get_user_id_from_email(conn, token_data.email)
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")

        cursor.execute(
            """
            INSERT INTO attempts (user_id, quiz_id, score, total_questions)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (user_id, req.quiz_id, req.score, req.total_questions)
        )
        attempt_id = cursor.fetchone()[0]
        conn.commit()
        return {"status": "success", "attempt_id": str(attempt_id)}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()

