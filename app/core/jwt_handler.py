from jose import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from fastapi import HTTPException

SECRET_KEY = "abcdef12345678"
ALGORITHM = "HS256"
TOKEN_TIME = 60

class TokenData(BaseModel):
    email: str

def create_access_token(data: TokenData):
    to_encode = data.dict()
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_TIME)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decoder_token(token : str) -> TokenData:
    try:
        dump_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(**dump_token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")