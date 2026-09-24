from __future__ import annotations

from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import ProviderCredential


SUPPORTED_PROVIDERS = {
    "groq",
    "openai",
    "gemini",
}


def normalize_provider(
    provider: str,
) -> str:

    normalized = (
        provider
        .strip()
        .lower()
    )

    if normalized not in SUPPORTED_PROVIDERS:

        supported = ", ".join(
            sorted(
                SUPPORTED_PROVIDERS
            )
        )

        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {supported}"
        )

    return normalized


def _get_fernet() -> Fernet:

    encryption_key = (
        settings.PROVIDER_CREDENTIAL_ENCRYPTION_KEY
    )

    if not encryption_key:

        raise RuntimeError(
            "PROVIDER_CREDENTIAL_ENCRYPTION_KEY "
            "is not configured."
        )

    try:

        return Fernet(
            encryption_key.encode()
        )

    except Exception as exc:

        raise RuntimeError(
            "PROVIDER_CREDENTIAL_ENCRYPTION_KEY "
            "is not a valid Fernet key."
        ) from exc


def encrypt_api_key(
    api_key: str,
) -> str:

    normalized_key = (
        api_key.strip()
    )

    if not normalized_key:

        raise ValueError(
            "API key cannot be empty."
        )

    return (
        _get_fernet()
        .encrypt(
            normalized_key.encode()
        )
        .decode()
    )


def decrypt_api_key(
    encrypted_api_key: str,
) -> str:

    if not encrypted_api_key:

        raise ValueError(
            "Encrypted API key cannot be empty."
        )

    try:

        return (
            _get_fernet()
            .decrypt(
                encrypted_api_key.encode()
            )
            .decode()
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to decrypt provider API key."
        ) from exc


async def save_provider_api_key(
    db: AsyncSession,
    provider: str,
    api_key: str,
) -> ProviderCredential:

    normalized_provider = (
        normalize_provider(
            provider
        )
    )

    encrypted_key = encrypt_api_key(
        api_key
    )

    result = await db.execute(
        select(ProviderCredential)
        .where(
            ProviderCredential.provider
            == normalized_provider
        )
    )

    credential = (
        result.scalar_one_or_none()
    )

    if credential is None:

        credential = ProviderCredential(
            provider=normalized_provider,
            encrypted_api_key=encrypted_key,
            is_active=True,
        )

        db.add(
            credential
        )

    else:

        credential.encrypted_api_key = (
            encrypted_key
        )

        credential.is_active = True

    await db.commit()

    await db.refresh(
        credential
    )

    return credential


async def get_provider_api_key(
    db: AsyncSession,
    provider: str,
) -> str | None:

    normalized_provider = (
        normalize_provider(
            provider
        )
    )

    result = await db.execute(
        select(
            ProviderCredential
        )
        .where(
            ProviderCredential.provider
            == normalized_provider,
            ProviderCredential.is_active.is_(
                True
            ),
        )
    )

    credential = (
        result.scalar_one_or_none()
    )

    if credential is None:
        return None

    return decrypt_api_key(
        credential.encrypted_api_key
    )


async def disable_provider_api_key(
    db: AsyncSession,
    provider: str,
) -> bool:

    normalized_provider = (
        normalize_provider(
            provider
        )
    )

    result = await db.execute(
        select(
            ProviderCredential
        )
        .where(
            ProviderCredential.provider
            == normalized_provider
        )
    )

    credential = (
        result.scalar_one_or_none()
    )

    if credential is None:
        return False

    credential.is_active = False

    await db.commit()

    return True


async def get_provider_credential_status(
    db: AsyncSession,
) -> list[dict]:

    result = await db.execute(
        select(
            ProviderCredential
        )
        .order_by(
            ProviderCredential.provider.asc()
        )
    )

    credentials = (
        result.scalars().all()
    )

    configured = {
        credential.provider: credential
        for credential in credentials
    }

    return [
        {
            "provider": provider,
            "configured": (
                provider in configured
                and configured[
                    provider
                ].is_active
            ),
            "status": (
                "connected"
                if (
                    provider in configured
                    and configured[
                        provider
                    ].is_active
                )
                else "not_configured"
            ),
        }
        for provider in sorted(
            SUPPORTED_PROVIDERS
        )
    ]