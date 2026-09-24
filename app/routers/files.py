import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.file_processor import (
    SUPPORTED_EXTENSIONS,
)
from app.models import (
    UploadedFile,
    User,
)
from app.rag import (
    RAGRuntimeConfig,
    ingest_file,
)
from app.services.system_settings import (
    get_system_settings,
)


router = APIRouter(
    prefix="/files",
    tags=["Files"],
)


# ============================================================
# VALIDATE EXTENSION
# ============================================================

def validate_extension(
    filename: str,
) -> str:

    from pathlib import Path

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        supported = ", ".join(
            sorted(
                SUPPORTED_EXTENSIONS
            )
        )

        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported file types: {supported}"
        )

    return extension


# ============================================================
# VALIDATE FILE SIZE
# ============================================================

def validate_file_size(
    file_size: int,
    maximum_size_mb: int,
) -> None:

    maximum_size = (
        maximum_size_mb
        * 1024
        * 1024
    )

    if file_size > maximum_size:

        raise ValueError(
            "File exceeds the maximum allowed size "
            f"of {maximum_size_mb} MB."
        )


# ============================================================
# GET USER FILES
# ============================================================

@router.get(
    "",
    response_model=list[dict],
)
async def list_files(
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.user_id
            == current_user.id
        )
        .order_by(
            UploadedFile.created_at.desc()
        )
    )

    uploaded_files = (
        result.scalars().all()
    )

    return [
        {
            "id": uploaded_file.id,
            "filename": (
                uploaded_file.original_filename
            ),
            "file_type": (
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
        }
        for uploaded_file in uploaded_files
    ]


# ============================================================
# UPLOAD FILE
# ============================================================

@router.post(
    "/upload",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    # ========================================================
    # LOAD CURRENT RUNTIME CONFIG
    # ========================================================

    system_settings = (
        await get_system_settings(
            db
        )
    )

    rag_config = RAGRuntimeConfig(
        chunk_size=(
            system_settings.chunk_size
        ),
        chunk_overlap=(
            system_settings.chunk_overlap
        ),
        retrieval_top_k=(
            system_settings.retrieval_top_k
        ),
        relevance_threshold=(
            system_settings.relevance_threshold
        ),
    )

    # ========================================================
    # VALIDATE FILENAME
    # ========================================================

    if not file.filename:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Filename is required.",
        )

    # ========================================================
    # VALIDATE EXTENSION
    # ========================================================

    try:

        extension = validate_extension(
            file.filename
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    # ========================================================
    # READ FILE
    # ========================================================

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Uploaded file is empty.",
        )

    # ========================================================
    # VALIDATE RUNTIME FILE SIZE
    # ========================================================

    try:

        validate_file_size(
            file_size=len(file_bytes),
            maximum_size_mb=(
                system_settings.max_file_size_mb
            ),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            ),
            detail=str(exc),
        ) from exc

    # ========================================================
    # MEMORY MARKER
    # ========================================================

    memory_marker = (
        f"memory://{uuid.uuid4().hex}"
    )

    # ========================================================
    # CREATE DATABASE METADATA
    # ========================================================

    uploaded_file = UploadedFile(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_filename=memory_marker,
        file_extension=extension,
        file_size=len(file_bytes),
        processing_status="processing",
        extracted_characters=0,
    )

    db.add(
        uploaded_file
    )

    await db.commit()

    await db.refresh(
        uploaded_file
    )

    # ========================================================
    # INGEST
    # ========================================================

    try:

        result = ingest_file(
            filename=file.filename,
            file_bytes=file_bytes,
            user_id=current_user.id,
            file_id=uploaded_file.id,
            max_zip_files=(
                system_settings.max_zip_files
            ),
            max_zip_uncompressed_size=(
                system_settings
                .max_zip_uncompressed_size_mb
                * 1024
                * 1024
            ),
            rag_config=rag_config,
        )

        uploaded_file.processing_status = (
            "processed"
        )

        uploaded_file.extracted_characters = (
            result[
                "extracted_characters"
            ]
        )

        uploaded_file.error_message = None

        await db.commit()

        await db.refresh(
            uploaded_file
        )

    except Exception as exc:

        uploaded_file.processing_status = (
            "failed"
        )

        uploaded_file.error_message = (
            str(exc)[:1000]
        )

        await db.commit()

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "The file could not be processed. "
                f"Reason: {str(exc)}"
            ),
        ) from exc

    finally:

        del file_bytes

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "id": uploaded_file.id,
        "filename": (
            uploaded_file.original_filename
        ),
        "file_type": (
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
        "chunks_created": (
            result["chunks"]
        ),
        "chunk_size": (
            result["chunk_size"]
        ),
        "chunk_overlap": (
            result["chunk_overlap"]
        ),
    }


# ============================================================
# DELETE FILE
# ============================================================

@router.delete(
    "/{file_id}",
    response_model=dict,
)
async def delete_file(
    file_id: int,
    current_user: User = Depends(
        get_current_user
    ),
    db: AsyncSession = Depends(
        get_db
    ),
):

    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.id == file_id,
            UploadedFile.user_id == current_user.id,
        )
    )

    uploaded_file = (
        result.scalar_one_or_none()
    )

    if uploaded_file is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Document not found.",
        )

    filename = (
        uploaded_file.original_filename
    )

    try:

        from app.rag import (
            delete_file_from_chroma,
        )

        deleted_chunks = (
            delete_file_from_chroma(
                file_id=file_id
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "The document could not be removed "
                "from the search index. "
                f"Reason: {str(exc)}"
            ),
        ) from exc

    await db.delete(
        uploaded_file
    )

    await db.commit()

    return {
        "success": True,
        "message": (
            "Document deleted successfully."
        ),
        "id": file_id,
        "filename": filename,
        "chunks_deleted": deleted_chunks,
    }