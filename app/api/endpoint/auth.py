from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class AuthLoginRequest(BaseModel):
    username: str
    password: str
class AuthRegisterRequest(BaseModel):
    email: str
    username: str
    password: str

@router.post("/login")
def auth_login(req : AuthLoginRequest):
    if req.username == "admin" and req.password == "admin":
        return {"status": "pass"}
    return {"status": "fail"}
    

@router.post("/register")
def auth_register(req : AuthRegisterRequest):
    return {"status": "pass"}

