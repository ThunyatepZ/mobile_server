from jose import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from fastapi import HTTPException
import os

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
TOKEN_TIME = 600

class TokenData(BaseModel):
    email: str

def create_access_token(data: TokenData):
    payload = data.dict()
    expires_at = datetime.utcnow() + timedelta(minutes=TOKEN_TIME)
    payload.update({"exp": expires_at})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decoder_token(token: str) -> TokenData:
    try:
        decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(**decoded_payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")