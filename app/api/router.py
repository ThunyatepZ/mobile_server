from app.api.endpoint import auth, course, chatapi
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth")
api_router.include_router(course.router, prefix="/courses")
api_router.include_router(chatapi.router, prefix="/chat")
