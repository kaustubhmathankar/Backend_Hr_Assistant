from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text

from app.database import (
    AsyncSessionLocal,
    engine,
)

from app.models import Base

# Import RBAC models so SQLAlchemy registers the
# roles / permissions / role_permissions tables.
import app.rbac_models  # noqa: F401

from app.graph import (
    initialize_graph,
    close_checkpointer,
)

from app.routers.admin_users import (
    router as admin_users_router,
)

from app.routers import admin
from app.routers import auth
from app.routers import chat
from app.routers import conversations
from app.routers import files
from app.routers import roles

from app.services.rbac import (
    initialize_rbac,
)


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "\n============================================================"
    )

    print(
        "STARTING VALETHI HR ASSISTANT"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Application database
    # --------------------------------------------------------

    async with engine.begin() as connection:

        await connection.run_sync(
            Base.metadata.create_all
        )

        # ----------------------------------------------------
        # EMAIL VERIFICATION COLUMN
        #
        # Existing users are treated as verified.
        # New registrations explicitly set this to False.
        # ----------------------------------------------------

        await connection.execute(
            text(
                """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS
                email_verified BOOLEAN
                NOT NULL DEFAULT TRUE
                """
            )
        )

    print(
        "[DATABASE] Application tables verified."
    )

    # --------------------------------------------------------
    # RBAC initialization
    # --------------------------------------------------------

    async with AsyncSessionLocal() as rbac_db:

        await initialize_rbac(
            rbac_db
        )

    print(
        "[RBAC] Roles and permissions initialized."
    )

    print(
        "[DATABASE] Conversation persistence enabled."
    )

    # --------------------------------------------------------
    # LangGraph PostgreSQL checkpointer
    # --------------------------------------------------------

    await initialize_graph()

    print(
        "[LANGGRAPH] PostgreSQL persistence enabled."
    )

    # --------------------------------------------------------
    # Application running
    # --------------------------------------------------------

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    await close_checkpointer()

    await engine.dispose()

    print(
        "\n============================================================"
    )

    print(
        "SHUTTING DOWN VALETHI HR ASSISTANT"
    )

    print(
        "============================================================"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Valethi HR Assistant",
    description=(
        "Production-grade AI HR Assistant using "
        "FastAPI, LangChain, LangGraph, RAG and PostgreSQL."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    admin_users_router
)

app.include_router(
    auth.router,
)

app.include_router(
    admin.router,
)

app.include_router(
    roles.router,
)

app.include_router(
    files.router,
)

app.include_router(
    chat.router,
)

app.include_router(
    conversations.router,
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Valethi HR Assistant API is running.",
        "version": "2.0.0",
        "langgraph_persistence": "postgresql",
    }