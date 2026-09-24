from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


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


def _common_kwargs(
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:

    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": 2,
    }


def create_llm(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> BaseChatModel:
    """
    Create a provider-specific LangChain chat model.

    Model IDs are deliberately NOT maintained in a static list.
    The supplied model must be validated by a real provider
    connection test before being saved as active configuration.
    """

    normalized_provider = (
        normalize_provider(
            provider
        )
    )

    normalized_model = (
        model.strip()
    )

    normalized_api_key = (
        api_key.strip()
    )

    if not normalized_model:

        raise ValueError(
            "Model cannot be empty."
        )

    if not normalized_api_key:

        raise ValueError(
            "API key cannot be empty."
        )

    if not 0 <= temperature <= 2:

        raise ValueError(
            "Temperature must be between 0 and 2."
        )

    if max_tokens <= 0:

        raise ValueError(
            "max_tokens must be greater than 0."
        )

    kwargs = _common_kwargs(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if normalized_provider == "groq":

        return ChatGroq(
            model=normalized_model,
            api_key=normalized_api_key,
            **kwargs,
        )

    if normalized_provider == "openai":

        return ChatOpenAI(
            model=normalized_model,
            api_key=normalized_api_key,
            **kwargs,
        )

    if normalized_provider == "gemini":

        return ChatGoogleGenerativeAI(
            model=normalized_model,
            google_api_key=normalized_api_key,
            **kwargs,
        )

    raise RuntimeError(
        "Unsupported provider."
    )


async def test_llm_connection(
    provider: str,
    model: str,
    api_key: str,
) -> dict:

    llm = create_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=0,
        max_tokens=32,
    )

    response = await llm.ainvoke(
        "Reply with exactly: CONNECTION_OK"
    )

    content = getattr(
        response,
        "content",
        "",
    )

    if isinstance(
        content,
        list,
    ):

        content = "".join(
            str(item)
            for item in content
        )

    content = str(
        content
    ).strip()

    return {
        "success": True,
        "provider": normalize_provider(
            provider
        ),
        "model": model.strip(),
        "response": content,
    }