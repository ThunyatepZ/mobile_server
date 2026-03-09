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
    answer = ask_chatbot(session_id, request.question)
    return {"answer": answer}


@router.post("/ask-upload")
async def ask_question_with_upload(
    question: str = Form(...),
    file: UploadFile | None = File(default=None),
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

    file_bytes = None
    filename = None
    if file is not None:
        file_bytes = await file.read()
        filename = file.filename or "uploaded_file"

    try:
        answer = ask_chatbot(
            session_id=session_id,
            question=question,
            uploaded_file_bytes=file_bytes,
            uploaded_filename=filename,
        )
        return {"answer": answer}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
