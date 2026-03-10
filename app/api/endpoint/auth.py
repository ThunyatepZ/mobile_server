from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
import bcrypt
from app.core.jwt_handler import TokenData, create_access_token, decoder_token
from app.db.superbase import get_db_connection

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AuthRegisterRequest(BaseModel):
    email: str
    username: str
    password: str


def get_db():
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    try:
        yield connection
    finally:
        connection.close()


def _hash_password(plain_password: str) -> str:
    hashed_bytes = bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt())
    return hashed_bytes.decode()


def _is_password_correct(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def _get_user_id_by_email(connection, email: str) -> Optional[str]:
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s;", (email,))
        row = cursor.fetchone()
        return str(row[0]) if row else None
    finally:
        cursor.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    conn=Depends(get_db),
):
    try:
        token_data = decoder_token(token)
        user_id = _get_user_id_by_email(conn, token_data.email)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user_id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/register")
def auth_register(req: AuthRegisterRequest, conn=Depends(get_db)):
    cursor = conn.cursor()

    try:
        hashed_password = _hash_password(req.password)

        cursor.execute(
            """
            INSERT INTO users (email, username, password)
            VALUES (%s, %s, %s)
            ON CONFLICT (email) DO NOTHING
            RETURNING id;
            """,
            (req.email, req.username, hashed_password),
        )

        row = cursor.fetchone() #row = array
        if not row:
            return {"status": "error", "message": "Email is already registered"}

        user_id = row[0]
        conn.commit()

        return {"status": "success", "user_id": str(user_id)}

    except Exception :
        conn.rollback()
        error_message = str(Exception)
        if "no unique or exclusion constraint matching the ON CONFLICT specification" in error_message:
            return {
                "status": "error",
                "message": (
                    "Database missing UNIQUE constraint on users.email (required for ON CONFLICT). "
                    "Please add a UNIQUE constraint/index on users(email)."
                ),
            }
        return {"status": "error", "message": error_message}

    finally:
        cursor.close()

@router.post("/login")
def auth_login(req: AuthLoginRequest, conn=Depends(get_db)):
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
        
        user_id, hashed_password = row
        if _is_password_correct(req.password, hashed_password):
            token = create_access_token(data=TokenData(email=req.email))
            return {
                "status": "success",
                "user_id": str(user_id),
                "access_token": token,
            }
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
        return {
            "status": "success",
            "user_id": str(user_id),
            "username": username,
            "email": email,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()

@router.get("/history")
def auth_history(
    token: str = Depends(oauth2_scheme),  # 1️⃣ ดึง Bearer token จาก Header
    conn = Depends(get_db)                # 2️⃣ ขอ database connection
):
    
    # 3️⃣ ถอดรหัส JWT token เพื่อเอาข้อมูล user (เช่น email)
    try:
        token_data = decoder_token(token)
    except HTTPException:
        # ถ้าเป็น HTTPException อยู่แล้ว → โยนต่อไปเลย
        raise
    except Exception:
        # ถ้า token ผิด format หรือหมดอายุ
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    # 4️⃣ สร้าง cursor เพื่อสั่ง SQL
    cursor = conn.cursor()

    try:
        # 5️⃣ Query ดึงประวัติการทำ quiz ของ user คนนี้
        cursor.execute(
            """
SELECT 
    a.id,                 -- attempt id
    q.id as quiz_id,      -- quiz id
    q.title as quiz_title,-- ชื่อ quiz
    a.score,              -- คะแนนที่ได้
    a.total_questions,    -- จำนวนข้อทั้งหมด
    a.completed_at        -- เวลาทำเสร็จ
FROM users u
JOIN attempts a ON u.id = a.user_id
JOIN quizzes q ON a.quiz_id = q.id
WHERE u.email = %s
ORDER BY a.completed_at DESC;
            """,
            (token_data.email,)  # ใช้ email จาก token
        )

        # 6️⃣ ดึงทุกแถวที่ได้จาก query
        rows = cursor.fetchall()

        # 7️⃣ เตรียม list สำหรับเก็บผลลัพธ์
        history = []

        # 8️⃣ วนลูปแต่ละแถวที่ได้จาก DB
        for attempt_row in rows:
            
            # unpack tuple
            attempt_id, quiz_id, quiz_title, score, total, completed_at = attempt_row

            # 9️⃣ แปลงเป็น dict แล้วเพิ่มเข้า list
            history.append(
                {
                    "attempt_id": str(attempt_id),
                    "quiz_id": str(quiz_id),
                    "quiz_title": quiz_title,
                    "score": score,
                    "total_questions": total,
                    "completed_at": str(completed_at),
                }
            )

        # 🔟 ส่งข้อมูลกลับไป frontend
        return {
            "status": "success",
            "history": history
        }

    except Exception as e:
        # ถ้า query พัง
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        # 1️⃣1️⃣ ปิด cursor ไม่ให้ memory leak
        cursor.close()