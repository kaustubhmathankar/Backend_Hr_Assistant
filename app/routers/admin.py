from __future__ import annotations

from collections import defaultdict

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
)

from pydantic import (
    BaseModel,
    Field,
)

from sqlalchemy import (
    func,
    select,
)

from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from app.auth import (
    get_current_user,
)

from app.database import (
    get_db,
)

from app.models import (
    Conversation,
    Message,
    UploadedFile,
    User,
)

from app.services.llm_factory import (
    SUPPORTED_PROVIDERS,
    test_llm_connection,
)

from app.services.model_catalog import (
    ModelCatalogError,
    get_available_models,
)

from app.services.provider_credentials import (
    disable_provider_api_key,
    get_provider_api_key,
    get_provider_credential_status,
    save_provider_api_key,
)

from app.services.system_settings import (
    get_system_settings,
    serialize_system_settings,
    update_system_settings,
)

from app.services.rbac import (
    has_permission,
    require_permission,
)


router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)


# ============================================================
# RBAC HELPERS
# ============================================================


async def require_any_permission(
    *,
    current_user: User,
    permission_keys: list[str],
    db: AsyncSession,
) -> None:

    for permission_key in permission_keys:

        allowed = await has_permission(
            current_user=current_user,
            permission_key=permission_key,
            db=db,
        )

        if allowed:
            return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "One of the following permissions is required: "
            + ", ".join(permission_keys)
        ),
    )


# ============================================================
# REQUEST MODELS
# ============================================================


class AIConnectionTestRequest(
    BaseModel
):
    """
    Request used to test a provider/model.

    api_key is optional because the backend can load the
    already-saved encrypted credential from PostgreSQL.
    """

    provider: str = Field(
        min_length=1,
        max_length=50,
    )

    model: str = Field(
        min_length=1,
        max_length=255,
    )

    api_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
    )


class AICredentialSaveRequest(
    BaseModel
):
    """
    Request used when saving a provider API key.

    The provider is already present in the URL, so the body
    only needs the API key.
    """

    api_key: str = Field(
        min_length=1,
        max_length=2000,
    )


class SystemSettingsUpdateRequest(
    BaseModel
):
    primary_provider: str | None = None
    primary_model: str | None = None

    utility_provider: str | None = None
    utility_model: str | None = None

    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
    )

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=200000,
    )

    chunk_size: int | None = Field(
        default=None,
        ge=100,
        le=50000,
    )

    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        le=20000,
    )

    retrieval_top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    relevance_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    max_file_size_mb: int | None = Field(
        default=None,
        ge=1,
        le=4096,
    )

    max_total_upload_size_mb: int | None = Field(
        default=None,
        ge=1,
        le=8192,
    )

    max_files_per_upload: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    max_zip_files: int | None = Field(
        default=None,
        ge=1,
        le=10000,
    )

    max_zip_uncompressed_size_mb: int | None = Field(
        default=None,
        ge=1,
        le=16384,
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================


@router.get(
    "/dashboard"
)
async def admin_dashboard(
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    allowed = await has_permission(
        current_user=current_admin,
        permission_key="dashboard.view",
        db=db,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                "dashboard.view"
            ),
        )


    # ========================================================
    # SYSTEM STATISTICS
    # ========================================================

    total_users_result = await db.execute(
        select(
            func.count(User.id)
        )
    )

    total_users = (
        total_users_result.scalar()
        or 0
    )


    total_documents_result = await db.execute(
        select(
            func.count(
                UploadedFile.id
            )
        )
    )

    total_documents = (
        total_documents_result.scalar()
        or 0
    )


    processed_documents_result = await db.execute(
        select(
            func.count(
                UploadedFile.id
            )
        ).where(
            UploadedFile.processing_status
            == "processed"
        )
    )

    processed_documents = (
        processed_documents_result.scalar()
        or 0
    )


    processing_documents_result = await db.execute(
        select(
            func.count(
                UploadedFile.id
            )
        ).where(
            UploadedFile.processing_status
            == "processing"
        )
    )

    processing_documents = (
        processing_documents_result.scalar()
        or 0
    )


    failed_documents_result = await db.execute(
        select(
            func.count(
                UploadedFile.id
            )
        ).where(
            UploadedFile.processing_status
            == "failed"
        )
    )

    failed_documents = (
        failed_documents_result.scalar()
        or 0
    )


    total_conversations_result = await db.execute(
        select(
            func.count(
                Conversation.id
            )
        )
    )

    total_conversations = (
        total_conversations_result.scalar()
        or 0
    )


    total_messages_result = await db.execute(
        select(
            func.count(
                Message.id
            )
        )
    )

    total_messages = (
        total_messages_result.scalar()
        or 0
    )


    # ========================================================
    # RECENT USERS
    # ========================================================

    users_result = await db.execute(
        select(User)
        .order_by(
            User.created_at.desc()
        )
        .limit(10)
    )

    recent_users = (
        users_result
        .scalars()
        .all()
    )


    users_data = [
        {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
        }
        for user in recent_users
    ]


    # ========================================================
    # RECENT DOCUMENTS
    # ========================================================

    documents_result = await db.execute(
        select(
            UploadedFile,
            User.email,
        )
        .join(
            User,
            User.id
            == UploadedFile.user_id,
        )
        .order_by(
            UploadedFile.created_at.desc()
        )
        .limit(10)
    )

    recent_documents = (
        documents_result
        .all()
    )


    documents_data = []


    for (
        uploaded_file,
        user_email,
    ) in recent_documents:

        documents_data.append(
            {
                "id": uploaded_file.id,
                "user_id": uploaded_file.user_id,
                "user_email": user_email,
                "filename": (
                    uploaded_file.original_filename
                ),
                "file_extension": (
                    uploaded_file.file_extension
                ),
                "file_size": (
                    uploaded_file.file_size
                ),
                "processing_status": (
                    uploaded_file.processing_status
                ),
                "extracted_characters": (
                    uploaded_file.extracted_characters
                ),
                "error_message": (
                    uploaded_file.error_message
                ),
                "created_at": (
                    uploaded_file.created_at
                ),
                "updated_at": (
                    uploaded_file.updated_at
                ),
            }
        )


    # ========================================================
    # RECENT CONVERSATIONS
    # ========================================================

    conversations_result = await db.execute(
        select(
            Conversation,
            User.email,
        )
        .join(
            User,
            User.id
            == Conversation.user_id,
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .limit(10)
    )

    recent_conversations = (
        conversations_result
        .all()
    )


    conversation_ids = [
        conversation.id
        for (
            conversation,
            _user_email,
        ) in recent_conversations
    ]


    message_counts_by_conversation = (
        defaultdict(int)
    )


    if conversation_ids:

        message_counts_result = await db.execute(
            select(
                Message.conversation_id,
                func.count(
                    Message.id
                ),
            )
            .where(
                Message.conversation_id.in_(
                    conversation_ids
                )
            )
            .group_by(
                Message.conversation_id
            )
        )


        for (
            conversation_id,
            message_count,
        ) in message_counts_result.all():

            message_counts_by_conversation[
                conversation_id
            ] = message_count


    conversation_messages_by_id = (
        defaultdict(list)
    )


    if conversation_ids:

        messages_result = await db.execute(
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


        messages = (
            messages_result
            .scalars()
            .all()
        )


        for message in messages:

            conversation_messages_by_id[
                message.conversation_id
            ].append(
                message
            )


    conversations_data = []


    for (
        conversation,
        user_email,
    ) in recent_conversations:

        messages = (
            conversation_messages_by_id.get(
                conversation.id,
                [],
            )
        )


        file_ids = []
        sources = []


        for message in messages:

            if isinstance(
                message.file_ids_json,
                list,
            ):

                for file_id in (
                    message.file_ids_json
                ):

                    try:

                        normalized_file_id = int(
                            file_id
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue


                    if (
                        normalized_file_id
                        not in file_ids
                    ):

                        file_ids.append(
                            normalized_file_id
                        )


            if isinstance(
                message.sources_json,
                list,
            ):

                for source in (
                    message.sources_json
                ):

                    if isinstance(
                        source,
                        str,
                    ):

                        source = (
                            source.strip()
                        )


                        if (
                            source
                            and source
                            not in sources
                        ):

                            sources.append(
                                source
                            )


                    elif isinstance(
                        source,
                        dict,
                    ):

                        source_name = (
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
                        )


                        if source_name:

                            source_name = str(
                                source_name
                            ).strip()


                            if (
                                source_name
                                and source_name
                                not in sources
                            ):

                                sources.append(
                                    source_name
                                )

        conversations_data.append(
            {
                "id": conversation.id,
                "session_id": (
                    conversation.session_id
                ),
                "user_id": (
                    conversation.user_id
                ),
                "user_email": user_email,
                "title": (
                    conversation.title
                    or "New Chat"
                ),
                "message_count": (
                    message_counts_by_conversation.get(
                        conversation.id,
                        0,
                    )
                ),
                "file_ids": file_ids,
                "sources": sources,
                "created_at": (
                    conversation.created_at
                ),
                "updated_at": (
                    conversation.updated_at
                ),
            }
        )


    # ========================================================
    # PIPELINE
    # ========================================================

    pipeline = {
        "ingestion": [
            {
                "id": "upload",
                "name": "Upload",
                "description": (
                    "User uploads a supported document."
                ),
            },
            {
                "id": "validate",
                "name": "Validate",
                "description": (
                    "Validate file type and file size."
                ),
            },
            {
                "id": "extract",
                "name": "Extract",
                "description": (
                    "Extract text from the uploaded document."
                ),
            },
            {
                "id": "chunk",
                "name": "Chunk",
                "description": (
                    "Split extracted text into searchable chunks."
                ),
            },
            {
                "id": "embed",
                "name": "Embed",
                "description": (
                    "Generate vector embeddings using the "
                    "configured embedding model."
                ),
            },
            {
                "id": "chroma",
                "name": "ChromaDB",
                "description": (
                    "Store document vectors and metadata."
                ),
            },
        ],

        "question_answering": [
            {
                "id": "question",
                "name": "User Question",
                "description": (
                    "Receive the authenticated user's question."
                ),
            },
            {
                "id": "rewrite",
                "name": "Query Rewrite",
                "description": (
                    "Prepare the query for retrieval."
                ),
            },
            {
                "id": "retrieve",
                "name": "Retrieve",
                "description": (
                    "Retrieve relevant chunks from ChromaDB."
                ),
            },
            {
                "id": "isolation",
                "name": "User/File Isolation",
                "description": (
                    "Restrict retrieval to the authenticated "
                    "user's selected files."
                ),
            },
            {
                "id": "relevance",
                "name": "Relevance Check",
                "description": (
                    "Check whether retrieved context is relevant."
                ),
            },
            {
                "id": "llm",
                "name": "Configured LLM",
                "description": (
                    "Generate the answer using the administrator's "
                    "active LLM provider and model."
                ),
            },
            {
                "id": "response",
                "name": "Response",
                "description": (
                    "Return the answer and document sources."
                ),
            },
        ],
    }


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "admin": {
            "id": current_admin.id,
            "email": current_admin.email,
            "role": current_admin.role,
            "is_active": current_admin.is_active,
        },

        "stats": {
            "total_users": total_users,
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "processing_documents": processing_documents,
            "failed_documents": failed_documents,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
        },

        "recent_users": users_data,

        "recent_documents": documents_data,

        "recent_conversations": conversations_data,

        "pipeline": pipeline,
    }


# ============================================================
# GET SYSTEM SETTINGS
# ============================================================


@router.get(
    "/settings"
)
async def get_admin_settings(
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    await require_any_permission(
        current_user=current_admin,
        permission_keys=[
            "ai.providers.view",
            "ai.runtime.view",
            "rag.settings.view",
        ],
        db=db,
    )


    system_settings = (
        await get_system_settings(
            db
        )
    )


    credentials = (
        await get_provider_credential_status(
            db
        )
    )


    return {
        "settings": (
            serialize_system_settings(
                system_settings
            )
        ),
        "providers": credentials,
    }


# ============================================================
# UPDATE SYSTEM SETTINGS
# ============================================================


@router.put(
    "/settings"
)
async def update_admin_settings(
    payload: SystemSettingsUpdateRequest,
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    values = payload.model_dump(
        exclude_none=True
    )


    if not values:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "At least one setting is required."
            ),
        )


    # --------------------------------------------------------
    # Determine which permission is required.
    # --------------------------------------------------------

    ai_runtime_fields = {
        "primary_provider",
        "primary_model",
        "utility_provider",
        "utility_model",
        "temperature",
        "max_tokens",
    }


    rag_setting_fields = {
        "chunk_size",
        "chunk_overlap",
        "retrieval_top_k",
        "relevance_threshold",
        "max_file_size_mb",
        "max_total_upload_size_mb",
        "max_files_per_upload",
        "max_zip_files",
        "max_zip_uncompressed_size_mb",
    }


    if any(
        field in values
        for field in ai_runtime_fields
    ):

        allowed = await has_permission(
            current_user=current_admin,
            permission_key="ai.runtime.manage",
            db=db,
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Permission required: "
                    "ai.runtime.manage"
                ),
            )


    if any(
        field in values
        for field in rag_setting_fields
    ):

        allowed = await has_permission(
            current_user=current_admin,
            permission_key="rag.settings.manage",
            db=db,
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Permission required: "
                    "rag.settings.manage"
                ),
            )


    # --------------------------------------------------------
    # Validate provider values.
    # --------------------------------------------------------

    for field_name in (
        "primary_provider",
        "utility_provider",
    ):

        provider = values.get(
            field_name
        )


        if provider is None:
            continue


        provider = (
            provider
            .strip()
            .lower()
        )


        if provider not in (
            SUPPORTED_PROVIDERS
        ):

            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"Unsupported provider "
                    f"'{provider}'."
                ),
            )


        values[
            field_name
        ] = provider


    # --------------------------------------------------------
    # Validate chunk relationship.
    # --------------------------------------------------------

    system_settings = (
        await get_system_settings(
            db
        )
    )


    chunk_size = values.get(
        "chunk_size",
        system_settings.chunk_size,
    )


    chunk_overlap = values.get(
        "chunk_overlap",
        system_settings.chunk_overlap,
    )


    if (
        chunk_overlap
        >= chunk_size
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Chunk overlap must be smaller "
                "than chunk size."
            ),
        )


    # --------------------------------------------------------
    # Update.
    # --------------------------------------------------------

    updated = (
        await update_system_settings(
            db=db,
            updated_by=current_admin.id,
            values=values,
        )
    )


    return {
        "success": True,
        "message": (
            "System settings updated successfully."
        ),
        "settings": (
            serialize_system_settings(
                updated
            )
        ),
    }


# ============================================================
# GET AI PROVIDERS
# ============================================================


@router.get(
    "/ai/providers"
)
async def get_ai_providers(
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    allowed = await has_permission(
        current_user=current_admin,
        permission_key="ai.providers.view",
        db=db,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                "ai.providers.view"
            ),
        )


    credentials = (
        await get_provider_credential_status(
            db
        )
    )


    return {
        "providers": credentials,
    }


# ============================================================
# GET LIVE PROVIDER MODELS
# ============================================================


@router.get(
    "/ai/models/{provider}"
)
async def get_ai_models(
    provider: str,
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    allowed = await has_permission(
        current_user=current_admin,
        permission_key="ai.models.view",
        db=db,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                "ai.models.view"
            ),
        )


    normalized_provider = (
        provider
        .strip()
        .lower()
    )


    if normalized_provider not in (
        SUPPORTED_PROVIDERS
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"Unsupported provider "
                f"'{normalized_provider}'."
            ),
        )


    api_key = (
        await get_provider_api_key(
            db=db,
            provider=normalized_provider,
        )
    )


    if not api_key:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"No API credential is configured "
                f"for provider '{normalized_provider}'. "
                "Save the provider API key before "
                "loading models."
            ),
        )


    try:

        models = (
            await get_available_models(
                provider=normalized_provider,
                api_key=api_key,
            )
        )

    except ModelCatalogError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
            ),
            detail=(
                f"Unable to load models from provider "
                f"'{normalized_provider}'. "
                f"Reason: {str(exc)}"
            ),
        ) from exc


    return {
        "success": True,
        "provider": normalized_provider,
        "count": len(models),
        "models": models,
    }
# ============================================================
# TEST PROVIDER / MODEL
# ============================================================


@router.post(
    "/ai/test"
)
async def test_ai_configuration(
    payload: AIConnectionTestRequest,
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    allowed = await has_permission(
        current_user=current_admin,
        permission_key="ai.providers.manage",
        db=db,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                "ai.providers.manage"
            ),
        )


    normalized_provider = (
        payload.provider
        .strip()
        .lower()
    )


    if normalized_provider not in (
        SUPPORTED_PROVIDERS
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"Unsupported provider "
                f"'{normalized_provider}'."
            ),
        )


    model = (
        payload.model
        .strip()
    )


    if not model:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Model is required.",
        )


    # --------------------------------------------------------
    # API key selection
    #
    # 1. If the user supplied a temporary API key,
    #    test that key.
    #
    # 2. Otherwise load the encrypted credential
    #    saved in PostgreSQL.
    # --------------------------------------------------------

    api_key = (
        payload.api_key.strip()
        if payload.api_key
        else None
    )


    if not api_key:

        api_key = (
            await get_provider_api_key(
                db=db,
                provider=normalized_provider,
            )
        )


        if not api_key:

            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    f"No API credential is configured "
                    f"for provider '{normalized_provider}'. "
                    "Save a provider API key before testing."
                ),
            )


    # --------------------------------------------------------
    # Test.
    # --------------------------------------------------------

    try:

        result = (
            await test_llm_connection(
                provider=normalized_provider,
                model=model,
                api_key=api_key,
            )
        )


        return {
            "success": True,
            "message": (
                "Provider connection and model "
                "test succeeded."
            ),
            "provider": result[
                "provider"
            ],
            "model": result[
                "model"
            ],
            "response": result[
                "response"
            ],
        }


    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Provider/model test failed. "
                f"Reason: {str(exc)}"
            ),
        ) from exc


# ============================================================
# SAVE PROVIDER API KEY
# ============================================================


@router.put(
    "/ai/credentials/{provider}"
)
async def save_ai_credentials(
    provider: str,
    payload: AICredentialSaveRequest,
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    allowed = await has_permission(
        current_user=current_admin,
        permission_key="ai.credentials.manage",
        db=db,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                "ai.credentials.manage"
            ),
        )


    normalized_provider = (
        provider
        .strip()
        .lower()
    )


    if normalized_provider not in (
        SUPPORTED_PROVIDERS
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"Unsupported provider "
                f"'{normalized_provider}'."
            ),
        )


    api_key = (
        payload.api_key
        .strip()
    )


    if not api_key:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "API key cannot be empty."
            ),
        )


    # --------------------------------------------------------
    # Validate the credential by discovering real provider
    # models with the newly supplied key.
    # --------------------------------------------------------

    try:

        models = (
            await get_available_models(
                provider=normalized_provider,
                api_key=api_key,
            )
        )


    except ModelCatalogError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "API key validation failed. "
                "Credential was not saved. "
                f"Reason: {str(exc)}"
            ),
        ) from exc


    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
            ),
            detail=(
                "Unable to validate the provider "
                "credential. Credential was not saved. "
                f"Reason: {str(exc)}"
            ),
        ) from exc


    if not models:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"The provider '{normalized_provider}' "
                "returned no usable chat/text models "
                "for this credential. Credential was not saved."
            ),
        )


    # --------------------------------------------------------
    # Store encrypted credential only after validation.
    # --------------------------------------------------------

    credential = (
        await save_provider_api_key(
            db=db,
            provider=normalized_provider,
            api_key=api_key,
        )
    )


    return {
        "success": True,
        "message": (
            "Provider credential validated "
            "and saved successfully."
        ),
        "provider": credential.provider,
        "configured": True,
        "model_count": len(models),
    }


# ============================================================
# DISABLE PROVIDER CREDENTIAL
# ============================================================


@router.delete(
    "/ai/credentials/{provider}"
)
async def remove_ai_credentials(
    provider: str,
    force: bool = Query(False, description="If true and the provider is active, attempt to switch active settings to another configured provider before disabling."),
    current_admin: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    allowed = await has_permission(
        current_user=current_admin,
        permission_key="ai.credentials.manage",
        db=db,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Permission required: "
                "ai.credentials.manage"
            ),
        )


    system_settings = (
        await get_system_settings(
            db
        )
    )


    normalized_provider = (
        provider
        .strip()
        .lower()
    )


    active_providers = {
        system_settings.primary_provider,
        system_settings.utility_provider,
    }


    if normalized_provider in (
        active_providers
    ):

        if not force:
            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    "This provider is currently used "
                    "by the active system configuration. "
                    "Switch to another configured provider "
                    "before disabling its credential."
                ),
            )

        # If force is requested, ensure the caller has runtime manage permission
        allowed_runtime = await has_permission(
            current_user=current_admin,
            permission_key="ai.runtime.manage",
            db=db,
        )
        if not allowed_runtime:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "To force-disable a provider that is currently active, "
                    "ai.runtime.manage permission is required."
                ),
            )

        # Attempt to find an alternate configured provider to switch to.
        credentials = (
            await get_provider_credential_status(
                db
            )
        )

        alternate = None
        for entry in credentials:
            if (
                entry.get("provider") != normalized_provider
                and entry.get("configured")
            ):
                alternate = entry.get("provider")
                break

        if not alternate:
            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    "No alternate configured provider is available to switch to. "
                    "Configure another provider before force-disabling this one."
                ),
            )

        # Update system settings to use the alternate provider where necessary.
        values = {}
        if system_settings.primary_provider == normalized_provider:
            values["primary_provider"] = alternate
        if system_settings.utility_provider == normalized_provider:
            values["utility_provider"] = alternate

        if values:
            await update_system_settings(
                db=db,
                updated_by=current_admin.id,
                values=values,
            )


    removed = (
        await disable_provider_api_key(
            db=db,
            provider=normalized_provider,
        )
    )


    if not removed:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Provider credential not found."
            ),
        )


    return {
        "success": True,
        "message": (
            "Provider credential disabled."
        ),
        "provider": normalized_provider,
    }