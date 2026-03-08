from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.service.chaobot import ask_chatbot
from app.api.endpoint.auth import oauth2_scheme, decoder_token

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    # ใช้ Email ของผู้ใช้ที่ล็อกอินเป็น Session_ID ของฝั่ง Memory
    session_id = token_data.email 
    answer = ask_chatbot(session_id, request.question)
    return {"answer": answer}