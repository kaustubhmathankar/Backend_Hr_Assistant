from __future__ import annotations

import json
from collections import OrderedDict, defaultdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import Conversation, Message, UploadedFile


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


# ============================================================
# HELPERS
# ============================================================

def normalize_file(
    file: UploadedFile,
) -> dict:
    """
    Convert an UploadedFile SQLAlchemy object into
    a JSON-friendly structure for the frontend.
    """

    return {
        "id": file.id,
        "filename": file.original_filename,
        "original_filename": file.original_filename,
        "file_extension": file.file_extension,
        "file_size": file.file_size,
        "processing_status": file.processing_status,
        "extracted_characters": file.extracted_characters,
        "error_message": file.error_message,
        "created_at": file.created_at,
    }


def normalize_list_value(
    value,
) -> list:
    """
    Convert a database JSON/JSONB value into a Python list.

    Handles:
        - list
        - tuple
        - JSON string such as "[18]"
        - None
    """

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    if isinstance(
        value,
        str,
    ):
        raw_value = value.strip()

        if not raw_value:
            return []

        try:
            decoded = json.loads(
                raw_value
            )

            if isinstance(
                decoded,
                list,
            ):
                return decoded

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return []

    return []


def extract_file_ids_from_messages(
    messages: list[Message],
) -> list[int]:
    """
    Extract file IDs from messages.

    IDs are:
        - converted to integers
        - deduplicated
        - kept in first-seen order
    """

    ordered_ids: OrderedDict[int, None] = (
        OrderedDict()
    )

    for message in messages:

        raw_file_ids = normalize_list_value(
            message.file_ids_json
        )

        for raw_file_id in raw_file_ids:

            try:
                file_id = int(
                    raw_file_id
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            ordered_ids.setdefault(
                file_id,
                None,
            )

    return list(
        ordered_ids.keys()
    )


async def load_messages_for_conversations(
    conversation_ids: list[int],
    db: AsyncSession,
) -> dict[int, list[Message]]:
    """
    Load messages directly from the Message table.

    Returns:

        {
            conversation_id: [
                Message,
                Message,
                ...
            ]
        }
    """

    if not conversation_ids:
        return {}

    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id.in_(
                conversation_ids
            )
        )
        .order_by(
            Message.created_at.asc()
        )
    )

    messages = result.scalars().all()

    grouped: dict[int, list[Message]] = defaultdict(
        list
    )

    for message in messages:
        grouped[
            message.conversation_id
        ].append(
            message
        )

    return dict(
        grouped
    )


async def load_files_for_conversation(
    file_ids: list[int],
    user_id: int,
    db: AsyncSession,
) -> list[UploadedFile]:
    """
    Load only files that belong to the authenticated user.

    The returned list follows the order of file_ids.
    """

    if not file_ids:
        return []

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id.in_(
                file_ids
            ),
            UploadedFile.user_id
            == user_id,
        )
    )

    files = result.scalars().all()

    files_by_id = {
        file.id: file
        for file in files
    }

    ordered_files = []

    for file_id in file_ids:

        file = files_by_id.get(
            file_id
        )

        if file is not None:
            ordered_files.append(
                file
            )

    return ordered_files


def normalize_sources(
    sources,
) -> list[str]:
    """
    Normalize persisted sources into a simple list
    of filename strings.
    """

    raw_sources = normalize_list_value(
        sources
    )

    normalized = []

    for source in raw_sources:

        if isinstance(
            source,
            str,
        ):

            value = source.strip()

            if value:
                normalized.append(
                    value
                )

            continue

        if isinstance(
            source,
            dict,
        ):

            value = (
                source.get(
                    "filename"
                )
                or source.get(
                    "file_name"
                )
                or source.get(
                    "source"
                )
                or source.get(
                    "name"
                )
                or ""
            )

            value = str(
                value
            ).strip()

            if value:
                normalized.append(
                    value
                )

    return list(
        OrderedDict.fromkeys(
            normalized
        )
    )


def build_message_response(
    message: Message,
) -> dict:
    """
    Convert a Message SQLAlchemy object into the
    frontend message format.
    """

    file_ids = normalize_list_value(
        message.file_ids_json
    )

    normalized_file_ids = []

    for raw_file_id in file_ids:

        try:
            normalized_file_ids.append(
                int(raw_file_id)
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "sources": normalize_sources(
            message.sources_json
        ),
        "file_ids": normalized_file_ids,
        "created_at": message.created_at,
    }


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

@router.get("")
async def list_conversations(
    current_user=Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):
    """
    Return all conversations belonging to
    the authenticated user.

    Each conversation contains:

        - title
        - message count
        - file_ids
        - files
    """

    # --------------------------------------------------------
    # Load conversations
    # --------------------------------------------------------

    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id
            == current_user.id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
    )

    conversations = (
        result.scalars().all()
    )

    # --------------------------------------------------------
    # Load messages directly
    # --------------------------------------------------------

    conversation_ids = [
        conversation.id
        for conversation in conversations
    ]

    messages_by_conversation = (
        await load_messages_for_conversations(
            conversation_ids,
            db,
        )
    )

    response = []

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    for conversation in conversations:

        messages = (
            messages_by_conversation.get(
                conversation.id,
                [],
            )
        )

        # -----------------------------------------------
        # Extract file IDs directly from Message records
        # -----------------------------------------------

        file_ids = (
            extract_file_ids_from_messages(
                messages
            )
        )

        # -----------------------------------------------
        # Load uploaded files
        # -----------------------------------------------

        uploaded_files = (
            await load_files_for_conversation(
                file_ids=file_ids,
                user_id=current_user.id,
                db=db,
            )
        )

        # -----------------------------------------------
        # Diagnostic logging
        # -----------------------------------------------

        print(
            "\n=================================================="
        )
        print(
            "[CONVERSATION]"
        )
        print(
            "=================================================="
        )

        print(
            f"Conversation ID: {conversation.id}"
        )

        print(
            f"Session ID: {conversation.session_id}"
        )

        print(
            f"Messages loaded: {len(messages)}"
        )

        print(
            f"Extracted file IDs: {file_ids}"
        )

        print(
            "Resolved files:"
        )

        for file in uploaded_files:
            print(
                f"  ID={file.id} | "
                f"NAME={file.original_filename} | "
                f"USER_ID={file.user_id} | "
                f"STATUS={file.processing_status}"
            )

        # -----------------------------------------------
        # Final response
        # -----------------------------------------------

        response.append(
            {
                "id": conversation.id,

                "session_id": (
                    conversation.session_id
                ),

                "title": (
                    conversation.title
                    or "New Chat"
                ),

                "created_at": (
                    conversation.created_at
                ),

                "updated_at": (
                    conversation.updated_at
                ),

                "message_count": (
                    len(messages)
                ),

                # IMPORTANT:
                # Return extracted IDs directly,
                # even if a file record cannot be resolved.
                "file_ids": file_ids,

                "files": [
                    normalize_file(
                        file
                    )
                    for file in uploaded_files
                ],
            }
        )

    return response


# ============================================================
# GET ONE CONVERSATION
# ============================================================

@router.get("/{session_id}")
async def get_conversation(
    session_id: str,
    current_user=Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):
    """
    Return one complete conversation.

    Includes:

        - conversation metadata
        - messages
        - message file IDs
        - message sources
        - associated uploaded files
    """

    session_id = session_id.strip()

    if not session_id:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required.",
        )

    # --------------------------------------------------------
    # Load conversation
    # --------------------------------------------------------

    result = await db.execute(
        select(Conversation).where(
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

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    # --------------------------------------------------------
    # Load messages directly
    # --------------------------------------------------------

    message_result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id
            == conversation.id
        )
        .order_by(
            Message.created_at.asc()
        )
    )

    messages = (
        message_result.scalars().all()
    )

    # --------------------------------------------------------
    # Extract file IDs
    # --------------------------------------------------------

    file_ids = (
        extract_file_ids_from_messages(
            messages
        )
    )

    # --------------------------------------------------------
    # Resolve files
    # --------------------------------------------------------

    uploaded_files = (
        await load_files_for_conversation(
            file_ids=file_ids,
            user_id=current_user.id,
            db=db,
        )
    )

    # --------------------------------------------------------
    # Normalize messages
    # --------------------------------------------------------

    normalized_messages = [
        build_message_response(
            message
        )
        for message in messages
    ]

    # --------------------------------------------------------
    # Diagnostic logging
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )
    print(
        "[CONVERSATION DETAIL]"
    )
    print(
        "=================================================="
    )

    print(
        f"Conversation ID: {conversation.id}"
    )

    print(
        f"Session ID: {conversation.session_id}"
    )

    print(
        f"Messages loaded: {len(messages)}"
    )

    print(
        f"Extracted file IDs: {file_ids}"
    )

    print(
        "Resolved files:"
    )

    for file in uploaded_files:
        print(
            f"  ID={file.id} | "
            f"NAME={file.original_filename} | "
            f"USER_ID={file.user_id} | "
            f"STATUS={file.processing_status}"
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "id": conversation.id,

        "session_id": (
            conversation.session_id
        ),

        "title": (
            conversation.title
            or "New Chat"
        ),

        "created_at": (
            conversation.created_at
        ),

        "updated_at": (
            conversation.updated_at
        ),

        "message_count": len(
            normalized_messages
        ),

        "file_ids": file_ids,

        "files": [
            normalize_file(
                file
            )
            for file in uploaded_files
        ],

        "messages": normalized_messages,
    }


# ============================================================
# DELETE CONVERSATION
# ============================================================

@router.delete("/{session_id}")
async def delete_conversation(
    session_id: str,
    current_user=Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):
    """
    Delete only the authenticated user's
    conversation.

    UploadedFile records are preserved.
    """

    session_id = session_id.strip()

    if not session_id:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required.",
        )

    result = await db.execute(
        select(Conversation).where(
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

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    await db.delete(
        conversation
    )

    await db.commit()

    return {
        "success": True,
        "session_id": session_id,
        "message": (
            "Conversation deleted successfully."
        ),
    }