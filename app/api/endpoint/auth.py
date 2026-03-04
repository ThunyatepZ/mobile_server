from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import bcrypt
from app.core.jwt_handler import create_access_token, TokenData, decoder_token
from app.db.superbase import get_db_connection

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AuthRegisterRequest(BaseModel):
    email: str
    username: str
    password: str


# Dependency
def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


@router.post("/register")
def auth_register(req: AuthRegisterRequest, conn=Depends(get_db)):
    cursor = conn.cursor()

    try:
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s;", (req.email,))
        if cursor.fetchone():
            return {"status": "error", "message": "Email is already registered"}

        # 🔐 hash password
        hashed_pw = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt())

        cursor.execute(
            """
            INSERT INTO users (email, username, password)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (req.email, req.username, hashed_pw.decode())
        )

        user_id = cursor.fetchone()[0]
        conn.commit()

        return {
            "status": "success",
            "user_id": str(user_id)
        }

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}

    finally:
        cursor.close()

@router.post("/login")
def auth_login(req : AuthLoginRequest, conn=Depends(get_db)):
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, password FROM users WHERE email = %s;
            """,
            (req.email,)
        )
        row = cursor.fetchone()
        if not row:
            return {"status": "error", "message": "User not found"}
        
        user_id, hashed_pw = row
        if bcrypt.checkpw(req.password.encode(), hashed_pw.encode()):
            token = create_access_token(data=TokenData(email=req.email))
            return {"status": "success", "user_id": str(user_id), "access_token": token}
        else:
            return {"status": "error", "message": "Invalid password"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        
        cursor.close()

@router.get("/me")
def auth_me(token: str = Depends(oauth2_scheme), conn=Depends(get_db)):
    try:
        token_data = decoder_token(token)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, username, email FROM users WHERE email = %s;
            """,
            (token_data.email,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        user_id, username, email = row
        return {"status": "success", "user_id": str(user_id), "username": username, "email": email}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()

@router.get("/history")
def auth_history(token: str = Depends(oauth2_scheme), conn=Depends(get_db)):
    try:
        token_data = decoder_token(token)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT c."courseID", c."courseName", e.id as enroll_id
            FROM users u
            JOIN enrollments e ON u.id = e.user_id
            JOIN courses c ON e.course_id = c.course_id
            WHERE u.email = %s;
            """,
            (token_data.email,)
        )
        rows = cursor.fetchall()
        
        history = []
        for row in rows:
            course_id, course_name, enroll_id = row
            history.append({
                "courseID": course_id,
                "courseName": course_name,
                "enrollment_id": str(enroll_id)
            })
            
        return {"status": "success", "history": history}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()