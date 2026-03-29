from app.api.endpoint import auth, chatapi, quiz, learning_path
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth")
api_router.include_router(chatapi.router, prefix="/chat")
api_router.include_router(quiz.router, prefix="/quiz")
api_router.include_router(learning_path.router, prefix="/learning-path")
