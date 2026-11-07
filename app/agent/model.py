import logging

from langchain_core.messages import AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_models = None


def _token_usage(response) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        response_metadata = getattr(response, "response_metadata", {})
        usage = (
            response_metadata.get("token_usage")
            or response_metadata.get("usage")
            if isinstance(response_metadata, dict)
            else None
        )
    if not isinstance(usage, dict):
        return None, None, None

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _log_token_usage(provider, response, request_id: str) -> None:
    usage = _token_usage(response)
    logger.info(
        "Model token usage conversation_id=%s provider=%s input_tokens=%s "
        "output_tokens=%s total_tokens=%s",
        request_id or "-",
        type(getattr(provider, "bound", provider)).__name__,
        *(value if value is not None else "unavailable" for value in usage),
    )


def llm():
    global _models
    if _models is None:
        configured_models = []
        if settings.groq_api_key:
            configured_models.append(
                ChatGroq(
                    model=settings.groq_model,
                    temperature=0,
                    api_key=settings.groq_api_key,
                    reasoning_format="parsed",
                    max_tokens=2048,
                )
            )
        if settings.gemini_api_key:
            configured_models.append(
                ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    temperature=0,
                    api_key=settings.gemini_api_key,
                )
            )
        if settings.openai_api_key:
            configured_models.append(
                ChatOpenAI(
                    model=settings.openai_model,
                    api_key=settings.openai_api_key,
                )
            )
        if not configured_models:
            raise RuntimeError(
                "No chat model API key is configured; set GROQ_API_KEY, "
                "GEMINI_API_KEY, or OPENAI_API_KEY"
            )
        _models = configured_models
    return _models[0]


def fallback_llms():
    return _models[1:] if _models is not None else []


def content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def invoke_with_fallbacks(models, messages, request_id: str = ""):
    failures = []
    for model in models:
        try:
            response = model.invoke(messages)
            _log_token_usage(model, response, request_id)
            return response
        except Exception as exc:
            provider = type(model).__name__
            failures.append(f"{provider}: {type(exc).__name__}: {exc}")
            logger.warning(
                "Chat model %s failed (%s: %s); trying the next configured provider",
                provider,
                type(exc).__name__,
                exc,
            )
    raise RuntimeError(
        f"All {len(models)} configured chat model provider(s) failed: "
        + " | ".join(failures)
    )


def stream_with_fallbacks(models, messages, writer, request_id: str = ""):
    failures = []
    for model in models:
        response: AIMessageChunk | None = None
        output_started = False
        try:
            for chunk in model.stream(messages):
                text = content_text(chunk.content)
                if text:
                    output_started = True
                    writer({"type": "token", "text": text})
                response = chunk if response is None else response + chunk
            if response is None:
                raise RuntimeError("The model returned no response chunks")
            _log_token_usage(model, response, request_id)
            return response
        except Exception as exc:
            if output_started:
                raise RuntimeError(
                    f"Chat model {type(model).__name__} streaming failed after "
                    "output began; refusing to retry to avoid duplicating a "
                    f"partial response: {type(exc).__name__}: {exc}"
                ) from exc
            provider = type(model).__name__
            failures.append(f"{provider}: {type(exc).__name__}: {exc}")
            logger.warning(
                "Chat model %s failed before streaming output (%s: %s); "
                "trying the next configured provider",
                provider,
                type(exc).__name__,
                exc,
            )
    raise RuntimeError(
        f"All {len(models)} configured chat model provider(s) failed: "
        + " | ".join(failures)
    )
