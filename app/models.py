from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database import Base


# ============================================================
# USER
# ============================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="user",
        server_default="user",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    # --------------------------------------------------------
    # EMAIL VERIFICATION
    #
    # Existing users are treated as verified.
    # New registrations explicitly set this to False.
    # --------------------------------------------------------

    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    password_reset_tokens: Mapped[
        list["PasswordResetToken"]
    ] = relationship(
        "PasswordResetToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    email_verification_tokens: Mapped[
        list["EmailVerificationToken"]
    ] = relationship(
        "EmailVerificationToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    uploaded_files: Mapped[
        list["UploadedFile"]
    ] = relationship(
        "UploadedFile",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    conversations: Mapped[
        list["Conversation"]
    ] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    updated_system_settings: Mapped[
        list["SystemSetting"]
    ] = relationship(
        "SystemSetting",
        back_populates="updated_by_user",
    )


# ============================================================
# PASSWORD RESET TOKEN
# ============================================================

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    token: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="password_reset_tokens",
    )


# ============================================================
# EMAIL VERIFICATION TOKEN
# ============================================================

class EmailVerificationToken(Base):
    __tablename__ = "email_verification_tokens"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    token: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="email_verification_tokens",
    )


# ============================================================
# UPLOADED FILE
# ============================================================

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    stored_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    file_extension: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    processing_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="processing",
        server_default="processing",
    )

    extracted_characters: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="uploaded_files",
    )


# ============================================================
# CONVERSATION
# ============================================================

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Chat",
        server_default="New Chat",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="conversations",
    )

    messages: Mapped[
        list["Message"]
    ] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


# ============================================================
# MESSAGE
# ============================================================

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    sources_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text(
            "'[]'::jsonb"
        ),
    )

    file_ids_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text(
            "'[]'::jsonb"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )


# ============================================================
# SYSTEM SETTINGS
# ============================================================

class SystemSetting(Base):
    """
    Runtime configuration controlled by administrators.

    Normally this table contains one active configuration row.
    The version field allows us to track configuration changes
    and later connect them to re-indexing and audit logs.
    """

    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    # --------------------------------------------------------
    # PRIMARY LLM
    # --------------------------------------------------------

    primary_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="groq",
        server_default="groq",
    )

    primary_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="openai/gpt-oss-20b",
        server_default="openai/gpt-oss-20b",
    )

    # --------------------------------------------------------
    # UTILITY LLM
    # --------------------------------------------------------

    utility_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="groq",
        server_default="groq",
    )

    utility_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="openai/gpt-oss-20b",
        server_default="openai/gpt-oss-20b",
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    temperature: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    max_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=2048,
        server_default="2048",
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    chunk_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1000,
        server_default="1000",
    )

    chunk_overlap: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=150,
        server_default="150",
    )

    retrieval_top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=5,
        server_default="5",
    )

    relevance_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.35,
        server_default="0.35",
    )

    # --------------------------------------------------------
    # UPLOAD LIMITS
    # --------------------------------------------------------

    max_file_size_mb: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=50,
        server_default="50",
    )

    max_total_upload_size_mb: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        server_default="100",
    )

    max_files_per_upload: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        server_default="10",
    )

    max_zip_files: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=100,
        server_default="100",
    )

    max_zip_uncompressed_size_mb: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=200,
        server_default="200",
    )

    # --------------------------------------------------------
    # EMBEDDING CONFIG
    # --------------------------------------------------------

    embedding_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        server_default=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    )

    # --------------------------------------------------------
    # CONFIG VERSION
    # --------------------------------------------------------

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )

    # --------------------------------------------------------
    # UPDATED BY
    # --------------------------------------------------------

    updated_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # --------------------------------------------------------
    # UPDATED AT
    # --------------------------------------------------------

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # --------------------------------------------------------
    # RELATIONSHIP
    # --------------------------------------------------------

    updated_by_user: Mapped["User | None"] = relationship(
        "User",
        back_populates="updated_system_settings",
        foreign_keys=[updated_by],
    )


# ============================================================
# PROVIDER CREDENTIAL
# ============================================================

class ProviderCredential(Base):
    """
    Stores an encrypted API credential for an LLM provider.

    The API key itself should never be returned through normal
    admin API responses.

    Example providers:

        groq
        openai
        gemini

    More providers can be added later.
    """

    __tablename__ = "provider_credentials"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
    )

    encrypted_api_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )