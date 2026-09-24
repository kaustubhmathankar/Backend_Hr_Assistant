from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
)


# ============================================================
# AUTH SCHEMAS
# ============================================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )


class UserResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    email: EmailStr
    role: str
    role_name: str | None = None
    is_active: bool
    email_verified: bool
    created_at: datetime
    permissions: list[str] = []


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str


class ResetPasswordRequest(BaseModel):
    token: str

    new_password: str = Field(
        min_length=8,
        max_length=128,
    )


class ResetPasswordResponse(BaseModel):
    message: str


class VerifyEmailResponse(BaseModel):
    message: str


# ============================================================
# FILE SCHEMAS
# ============================================================

class UploadedFileResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    original_filename: str
    stored_filename: str
    file_extension: str
    file_size: int
    processing_status: str
    extracted_characters: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ============================================================
# CHAT SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=10000,
    )

    file_ids: list[int] = Field(
        min_length=1,
    )

    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str] = []


# ============================================================
# CONVERSATION SCHEMAS
# ============================================================

class ConversationResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    conversation_id: int
    role: str
    content: str
    sources: list[str] = []
    file_ids: list[int] = []
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    id: int
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []