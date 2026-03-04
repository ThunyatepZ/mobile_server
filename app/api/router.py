from app.api.endpoint import auth, course
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth")
