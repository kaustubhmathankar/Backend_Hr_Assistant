from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SystemSetting


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_SYSTEM_SETTINGS = {
    "primary_provider": "groq",
    "primary_model": "openai/gpt-oss-20b",
    "utility_provider": "groq",
    "utility_model": "openai/gpt-oss-20b",
    "temperature": 0.0,
    "max_tokens": 2048,
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "retrieval_top_k": 5,
    "relevance_threshold": 0.35,
    "max_file_size_mb": 50,
    "max_total_upload_size_mb": 100,
    "max_files_per_upload": 10,
    "max_zip_files": 100,
    "max_zip_uncompressed_size_mb": 200,
    "embedding_model": (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    ),
    "version": 1,
}


# ============================================================
# GET ACTIVE SETTINGS
# ============================================================

async def get_system_settings(
    db: AsyncSession,
) -> SystemSetting:
    """
    Return the active system configuration.

    The application currently uses one active configuration
    row. If none exists, it is created automatically from
    DEFAULT_SYSTEM_SETTINGS.
    """

    result = await db.execute(
        select(SystemSetting)
        .order_by(
            SystemSetting.id.asc()
        )
        .limit(1)
    )

    system_settings = (
        result.scalar_one_or_none()
    )

    if system_settings is not None:
        return system_settings

    system_settings = SystemSetting(
        **DEFAULT_SYSTEM_SETTINGS
    )

    db.add(
        system_settings
    )

    await db.commit()

    await db.refresh(
        system_settings
    )

    return system_settings


# ============================================================
# UPDATE ACTIVE SETTINGS
# ============================================================

async def update_system_settings(
    db: AsyncSession,
    updated_by: int,
    values: dict,
) -> SystemSetting:
    """
    Update the active configuration.

    The configuration version increments whenever a successful
    update is committed.
    """

    system_settings = (
        await get_system_settings(
            db
        )
    )

    for field_name, value in values.items():

        if not hasattr(
            SystemSetting,
            field_name,
        ):
            continue

        setattr(
            system_settings,
            field_name,
            value,
        )

    system_settings.version += 1

    system_settings.updated_by = (
        updated_by
    )

    await db.commit()

    await db.refresh(
        system_settings
    )

    return system_settings


# ============================================================
# SERIALIZE PUBLIC SETTINGS
# ============================================================

def serialize_system_settings(
    system_settings: SystemSetting,
) -> dict:
    """
    Convert configuration to a safe API response.

    Provider API keys are intentionally not part of this
    response.
    """

    return {
        "id": system_settings.id,

        "primary_provider": (
            system_settings.primary_provider
        ),

        "primary_model": (
            system_settings.primary_model
        ),

        "utility_provider": (
            system_settings.utility_provider
        ),

        "utility_model": (
            system_settings.utility_model
        ),

        "temperature": (
            system_settings.temperature
        ),

        "max_tokens": (
            system_settings.max_tokens
        ),

        "chunk_size": (
            system_settings.chunk_size
        ),

        "chunk_overlap": (
            system_settings.chunk_overlap
        ),

        "retrieval_top_k": (
            system_settings.retrieval_top_k
        ),

        "relevance_threshold": (
            system_settings.relevance_threshold
        ),

        "max_file_size_mb": (
            system_settings.max_file_size_mb
        ),

        "max_total_upload_size_mb": (
            system_settings.max_total_upload_size_mb
        ),

        "max_files_per_upload": (
            system_settings.max_files_per_upload
        ),

        "max_zip_files": (
            system_settings.max_zip_files
        ),

        "max_zip_uncompressed_size_mb": (
            system_settings.max_zip_uncompressed_size_mb
        ),

        "embedding_model": (
            system_settings.embedding_model
        ),

        "version": (
            system_settings.version
        ),

        "updated_by": (
            system_settings.updated_by
        ),

        "updated_at": (
            system_settings.updated_at
        ),
    }