from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List
from app.db.superbase import get_db_connection
from app.api.endpoint.auth import oauth2_scheme, decoder_token

router = APIRouter()

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# Request Models
class EnrollRequest(BaseModel):
    course_id: str

@router.get("/")
def get_courses(conn=Depends(get_db)):
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, "courseID", "courseName", "courseNameEn", credits, description, "imageUrl", topics
            FROM courses;
            """
        )
        rows = cursor.fetchall()
        courses = []
        for row in rows:
            course = {
                "id": str(row[0]),
                "courseID": row[1],
                "courseName": row[2],
                "courseNameEn": row[3],
                "credits": row[4],
                "description": row[5],
                "imageUrl": row[6],
                "topics": row[7] if row[7] else []
            }
            courses.append(course)
        return {"status": "success", "courses": courses}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()

@router.post("/enroll")
def enroll_course(req: EnrollRequest, token: str = Depends(oauth2_scheme), conn=Depends(get_db)):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    cursor = conn.cursor()
    try:
        # Get user id
        cursor.execute("SELECT id FROM users WHERE email = %s;", (token_data.email,))
        user_row = cursor.fetchone()
        if not user_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        user_id = user_row[0]

        cursor.execute(
            """
            INSERT INTO enrollments (user_id, course_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id, course_id) DO NOTHING
            RETURNING id;
            """,
            (user_id, req.course_id)
        )
        enroll = cursor.fetchone()
        conn.commit()

        if not enroll:
            return {"status": "success", "message": "Already enrolled"}

        return {"status": "success", "message": "Enrolled successfully", "enrollment_id": str(enroll[0])}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()