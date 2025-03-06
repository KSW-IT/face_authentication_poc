from pydantic import BaseModel

class UserRegister(BaseModel):
    username: str
    password: str
    email: str


class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class ChangePassword(BaseModel):
    email: str
    currentPassword: str
    newPassword: str
class Response(BaseModel):
    code: str
    message: str
