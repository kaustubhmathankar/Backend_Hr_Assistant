from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db

from app.graph import run_chat

from app.models import (
    Conversation,
    Message,
    UploadedFile,
)

from app.schemas import (
    ChatRequest,
    ChatResponse,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# ============================================================
# CONVERSATION TITLE
# ============================================================

def generate_title(
    question: str,
) -> str:

    title = " ".join(
        question
        .strip()
        .split()
    )

    if len(title) <= 60:

        return title

    return (
        title[:57]
        .rstrip()
        + "..."
    )


# ============================================================
# CHAT ENDPOINT
# ============================================================

@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    current_user=Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    question = (
        request.question
        .strip()
    )

    if not question:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Question cannot be empty."
            ),
        )

    # ========================================================
    # VALIDATE FILE IDS
    # ========================================================

    if not request.file_ids:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "At least one uploaded file "
                "must be selected."
            ),
        )

    # --------------------------------------------------------
    # Remove duplicates while preserving order.
    # --------------------------------------------------------

    file_ids = list(
        dict.fromkeys(
            request.file_ids
        )
    )

    # ========================================================
    # LOAD SELECTED FILES
    # ========================================================

    result = await db.execute(
        select(
            UploadedFile
        ).where(
            UploadedFile.id.in_(
                file_ids
            ),
            UploadedFile.user_id
            == current_user.id,
        )
    )

    files = (
        result.scalars().all()
    )

    # ========================================================
    # VERIFY OWNERSHIP
    # ========================================================

    found_file_ids = {
        file.id
        for file in files
    }

    requested_file_ids = set(
        file_ids
    )

    invalid_file_ids = (
        requested_file_ids
        - found_file_ids
    )

    if invalid_file_ids:

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail={
                "message": (
                    "One or more selected files "
                    "do not belong to the current user."
                ),
                "invalid_file_ids": list(
                    invalid_file_ids
                ),
            },
        )

    # ========================================================
    # FILE STATUS
    # ========================================================

    READY_STATUSES = {
        "completed",
        "processed",
        "ready",
        "success",
    }

    PROCESSING_STATUSES = {
        "processing",
        "pending",
        "queued",
    }

    FAILED_STATUSES = {
        "failed",
        "error",
    }

    for file in files:

        if (
            file.processing_status
            in PROCESSING_STATUSES
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    f"Document "
                    f"'{file.original_filename}' "
                    f"is still processing."
                ),
            )

        if (
            file.processing_status
            in FAILED_STATUSES
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ),
                detail=(
                    f"Document "
                    f"'{file.original_filename}' "
                    f"could not be processed."
                ),
            )

        if (
            file.processing_status
            not in READY_STATUSES
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    f"Document "
                    f"'{file.original_filename}' "
                    f"is not ready for use."
                ),
            )

    # ========================================================
    # DIAGNOSTIC LOGGING
    # ========================================================

    print(
        "\n=================================================="
    )

    print(
        "[CHAT] Selected files"
    )

    print(
        "=================================================="
    )

    for file in files:

        print(
            f"ID={file.id} | "
            f"NAME={file.original_filename} | "
            f"STATUS={file.processing_status}"
        )

    # ========================================================
    # RESOLVE CONVERSATION
    # ========================================================

    session_id = (
        request.session_id
        if request.session_id
        else None
    )

    conversation = None

    if session_id:

        session_id = (
            session_id
            .strip()
        )

        if not session_id:

            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "Session ID cannot be empty."
                ),
            )

        result = await db.execute(
            select(
                Conversation
            ).where(
                Conversation.session_id
                == session_id,
                Conversation.user_id
                == current_user.id,
            )
        )

        conversation = (
            result.scalar_one_or_none()
        )

        if conversation is None:

            conversation = Conversation(
                session_id=session_id,
                user_id=current_user.id,
                title=generate_title(
                    question
                ),
            )

            db.add(
                conversation
            )

            await db.flush()

    else:

        session_id = str(
            uuid4()
        )

        conversation = Conversation(
            session_id=session_id,
            user_id=current_user.id,
            title=generate_title(
                question
            ),
        )

        db.add(
            conversation
        )

        await db.flush()

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=question,
        sources_json=[],
        file_ids_json=file_ids,
    )

    db.add(
        user_message
    )

    await db.flush()

    # ========================================================
    # RUN LANGGRAPH
    # ========================================================

    try:

        graph_result = await run_chat(
            db=db,
            user_id=current_user.id,
            question=question,
            file_ids=file_ids,
            session_id=session_id,
        )

    except ValueError as exc:

        await db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:

        await db.rollback()

        print(
            "\n=================================================="
        )

        print(
            "[CHAT CONFIGURATION ERROR]"
        )

        print(
            str(exc)
        )

        print(
            "=================================================="
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The AI provider is not configured "
                "correctly right now."
            ),
        ) from exc

    except Exception as exc:

        await db.rollback()

        print(
            "\n=================================================="
        )

        print(
            "[CHAT GRAPH ERROR]"
        )

        print(
            str(exc)
        )

        print(
            "=================================================="
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unable to process your question "
                "right now."
            ),
        ) from exc

    # ========================================================
    # GRAPH RESPONSE
    # ========================================================

    answer = str(
        graph_result.get(
            "answer",
            "",
        )
    ).strip()

    sources = graph_result.get(
        "sources",
        [],
    )

    if not answer:

        answer = (
            "I couldn't generate an answer "
            "for this question."
        )

    if not isinstance(
        sources,
        list,
    ):

        sources = []

    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources_json=sources,
        file_ids_json=file_ids,
    )

    db.add(
        assistant_message
    )

    # --------------------------------------------------------
    # Update conversation timestamp explicitly.
    # --------------------------------------------------------

    conversation.updated_at = (
        datetime.now(
            timezone.utc
        )
    )

    # ========================================================
    # COMMIT
    # ========================================================

    await db.commit()

    # ========================================================
    # RESPONSE
    # ========================================================

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
    )