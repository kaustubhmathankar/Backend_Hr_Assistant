from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.services.llm_factory import SUPPORTED_PROVIDERS


class ModelCatalogError(RuntimeError):
    """Raised when a provider model catalog cannot be loaded safely."""


REQUEST_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_GEMINI_PAGES = 10


# These are capability-family hints, not a static model catalog.
# Provider APIs remain the source of truth for actual model IDs.
_NON_TEXT_MODEL_HINTS = (
    "embedding",
    "moderation",
    "whisper",
    "transcribe",
    "translation",
    "tts",
    "text-to-speech",
    "image-generation",
    "image-edit",
    "dall-e",
    "realtime",
)


def _model_id(value: Any) -> str:
    return str(value or "").strip()


def _looks_like_non_text_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(hint in lowered for hint in _NON_TEXT_MODEL_HINTS)


def _normalize_model(
    *,
    provider: str,
    model_id: str,
    display_name: str | None = None,
    active: bool | None = None,
    context_window: int | None = None,
    owned_by: str | None = None,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "display_name": (display_name or model_id).strip(),
        "provider": provider,
        "active": active,
        "context_window": context_window,
        "owned_by": owned_by,
    }


def _parse_http_error(provider: str, response: httpx.Response) -> ModelCatalogError:
    detail = ""

    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = str(
                error.get("message")
                or error.get("detail")
                or ""
            ).strip()
        elif isinstance(error, str):
            detail = error.strip()

        if not detail:
            detail = str(payload.get("message") or payload.get("detail") or "").strip()

    if response.status_code in {401, 403}:
        return ModelCatalogError(
            f"The {provider} API credential was rejected while loading models."
        )

    if detail:
        return ModelCatalogError(
            f"The {provider} model service returned an error: {detail}"
        )

    return ModelCatalogError(
        f"The {provider} model service returned HTTP {response.status_code}."
    )


async def _get_json(
    client: httpx.AsyncClient,
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = await client.get(
            url,
            headers=headers,
            params=params,
        )
    except httpx.TimeoutException as exc:
        raise ModelCatalogError(
            f"Timed out while loading models from {provider}."
        ) from exc
    except httpx.RequestError as exc:
        raise ModelCatalogError(
            f"Unable to reach the {provider} model service."
        ) from exc

    if response.status_code >= 400:
        raise _parse_http_error(provider, response)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelCatalogError(
            f"The {provider} model service returned an invalid response."
        ) from exc

    if not isinstance(payload, dict):
        raise ModelCatalogError(
            f"The {provider} model service returned an unexpected response."
        )

    return payload


async def _fetch_groq_models(
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict[str, Any]]:
    payload = await _get_json(
        client,
        provider="Groq",
        url="https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    result: list[dict[str, Any]] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue

        model_id = _model_id(item.get("id"))
        if not model_id:
            continue

        if item.get("active") is False:
            continue

        if _looks_like_non_text_model(model_id):
            continue

        result.append(
            _normalize_model(
                provider="groq",
                model_id=model_id,
                display_name=model_id,
                active=item.get("active"),
                context_window=item.get("context_window"),
                owned_by=item.get("owned_by"),
            )
        )

    return _deduplicate_and_sort(result)


async def _fetch_openai_models(
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict[str, Any]]:
    payload = await _get_json(
        client,
        provider="OpenAI",
        url="https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    now = datetime.now(timezone.utc)
    result: list[dict[str, Any]] = []

    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue

        model_id = _model_id(item.get("id"))
        if not model_id:
            continue

        if _looks_like_non_text_model(model_id):
            continue

        shutdown_date = item.get("shutdown_date")
        if shutdown_date:
            try:
                parsed_shutdown = datetime.fromisoformat(
                    str(shutdown_date).replace("Z", "+00:00")
                )
                if parsed_shutdown.tzinfo is None:
                    parsed_shutdown = parsed_shutdown.replace(tzinfo=timezone.utc)
                if parsed_shutdown <= now:
                    continue
            except ValueError:
                # An unparseable shutdown date should not cause the entire
                # catalog to fail. The provider still returned the model.
                pass

        result.append(
            _normalize_model(
                provider="openai",
                model_id=model_id,
                display_name=model_id,
                active=True,
                owned_by=item.get("owned_by"),
            )
        )

    return _deduplicate_and_sort(result)


async def _fetch_gemini_models(
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    page_token: str | None = None

    for _ in range(MAX_GEMINI_PAGES):
        params: dict[str, Any] = {
            "pageSize": 100,
        }
        if page_token:
            params["pageToken"] = page_token

        payload = await _get_json(
            client,
            provider="Google Gemini",
            url="https://generativelanguage.googleapis.com/v1beta/models",
            headers={
                "x-goog-api-key": api_key,
                "Accept": "application/json",
            },
            params=params,
        )

        for item in payload.get("models") or []:
            if not isinstance(item, dict):
                continue

            raw_name = _model_id(item.get("name"))
            if not raw_name:
                continue

            model_id = raw_name.removeprefix("models/")
            if not model_id:
                continue

            supported_methods = item.get("supportedGenerationMethods") or []
            normalized_methods = {
                str(method).strip()
                for method in supported_methods
            }

            # Gemini explicitly exposes generation capabilities. This is the
            # strongest provider-side check for a model suitable for chat/text
            # generation.
            if "generateContent" not in normalized_methods:
                continue

            if _looks_like_non_text_model(model_id):
                continue

            result.append(
                _normalize_model(
                    provider="gemini",
                    model_id=model_id,
                    display_name=item.get("displayName") or model_id,
                    active=True,
                    context_window=item.get("inputTokenLimit"),
                )
            )

        page_token = _model_id(payload.get("nextPageToken")) or None
        if not page_token:
            break

    return _deduplicate_and_sort(result)


def _deduplicate_and_sort(
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}

    for model in models:
        model_id = _model_id(model.get("id"))
        if not model_id:
            continue
        unique[model_id] = model

    return sorted(
        unique.values(),
        key=lambda item: (
            not bool(item.get("active")),
            str(item.get("display_name") or item.get("id") or "").lower(),
            str(item.get("id") or "").lower(),
        ),
    )


async def get_available_models(
    *,
    provider: str,
    api_key: str,
) -> list[dict[str, Any]]:
    normalized_provider = str(provider or "").strip().lower()
    credential = str(api_key or "").strip()

    if normalized_provider not in SUPPORTED_PROVIDERS:
        raise ModelCatalogError(
            f"Unsupported provider '{normalized_provider}'."
        )

    if not credential:
        raise ModelCatalogError(
            f"An API credential is required for provider '{normalized_provider}'."
        )

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        if normalized_provider == "groq":
            return await _fetch_groq_models(client, credential)

        if normalized_provider == "openai":
            return await _fetch_openai_models(client, credential)

        if normalized_provider == "gemini":
            return await _fetch_gemini_models(client, credential)

    raise ModelCatalogError(
        f"No model catalog implementation exists for provider '{normalized_provider}'."
    )

