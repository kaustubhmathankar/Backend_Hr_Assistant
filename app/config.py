from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):

    # ========================================================
    # APPLICATION
    # ========================================================

    APP_NAME: str = "Valethi HR Assistant"

    # ========================================================
    # DATABASE
    # ========================================================

    DATABASE_URL: str

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ========================================================
    # EMAIL / RESEND
    # ========================================================

    RESEND_API_KEY: str = ""
    EMAIL_FROM: str = "onboarding@resend.dev"

    # ========================================================
    # LEGACY GROQ SETTINGS
    #
    # These remain temporarily so the current application
    # continues working during the provider-abstraction
    # migration.
    #
    # Later the active runtime values will come from
    # system_settings + provider_credentials.
    # ========================================================

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # ========================================================
    # OPENAI
    #
    # Optional during the migration.
    # ========================================================

    OPENAI_API_KEY: str = ""

    # ========================================================
    # GEMINI
    #
    # Optional during the migration.
    # ========================================================

    GOOGLE_API_KEY: str = ""

    # ========================================================
    # PROVIDER CREDENTIAL ENCRYPTION
    #
    # Required once provider credentials are stored in the
    # database.
    #
    # Generate a Fernet-compatible key for production.
    # ========================================================

    PROVIDER_CREDENTIAL_ENCRYPTION_KEY: str = ""

    # ========================================================
    # CHROMA
    # ========================================================

    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    CHROMA_COLLECTION_NAME: str = "hr_documents"

    # ========================================================
    # LEGACY UPLOAD SETTINGS
    #
    # These remain temporarily for compatibility while the
    # runtime configuration layer is introduced.
    # ========================================================

    UPLOAD_MAX_SIZE_MB: int = 50

    MAX_ZIP_FILES: int = 100

    MAX_ZIP_UNCOMPRESSED_SIZE_MB: int = 200

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    EMBEDDING_MODEL: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    # ========================================================
    # LEGACY RAG SETTINGS
    #
    # Runtime versions will eventually come from
    # system_settings.
    # ========================================================

    RAG_TOP_K: int = 5

    RAG_RELEVANCE_THRESHOLD: float = 0.35

    # ========================================================
    # SETTINGS CONFIGURATION
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()