from app.api.endpoint import auth, chatapi, quiz
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth")
api_router.include_router(chatapi.router, prefix="/chat")
api_router.include_router(quiz.router, prefix="/quiz")
