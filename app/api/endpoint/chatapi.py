from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.endpoint.auth import decoder_token, oauth2_scheme
from app.service.chaobot import ask_chatbot

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    # ใช้ Email ของผู้ใช้ที่ล็อกอินเป็น Session_ID ของฝั่ง Memory
    session_id = token_data.email
    result = ask_chatbot(session_id, request.question)
    return result


@router.post("/ask-upload")
async def ask_question_with_upload(
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    token: str = Depends(oauth2_scheme),
):
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    session_id = token_data.email

    uploaded_files_data = []
    for file in files:
        content = await file.read()
        uploaded_files_data.append({
            "bytes": content,
            "filename": file.filename or "uploaded_file"
        })

    try:
        result = ask_chatbot(
            session_id=session_id,
            question=question,
            uploaded_files=uploaded_files_data,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
