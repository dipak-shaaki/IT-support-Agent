import logging
from time import perf_counter

from langchain_core.messages import AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.metrics import (
    LLM_CALLS,
    LLM_CALL_DURATION,
    LLM_FAILURES,
    LLM_GENERATION_DURATION,
    LLM_OUTPUT_TOKENS,
    LLM_TOKENS_PER_SECOND,
    LLM_TTFT,
)

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


def _provider_name(provider) -> str:
    return type(getattr(provider, "bound", provider)).__name__


def _model_name(provider) -> str:
    bound = getattr(provider, "bound", provider)
    return str(
        getattr(bound, "model_name", None)
        or getattr(bound, "model", None)
        or "unknown"
    )


def _record_call(
    provider,
    mode: str,
    outcome: str,
    started: float,
    request_id: str,
    call_index: int,
    *,
    ttft: float | None = None,
    generation_seconds: float | None = None,
    response=None,
) -> dict:
    elapsed = perf_counter() - started
    provider_name = _provider_name(provider)
    usage = _token_usage(response) if response is not None else (None, None, None)
    input_tokens, output_tokens, total_tokens = usage
    tokens_per_second = (
        output_tokens / generation_seconds
        if output_tokens is not None and generation_seconds and generation_seconds > 0
        else None
    )

    LLM_CALLS.labels(provider_name, mode, outcome).inc()
    LLM_CALL_DURATION.labels(provider_name, mode).observe(elapsed)
    if ttft is not None:
        LLM_TTFT.labels(provider_name).observe(ttft)
    if generation_seconds is not None:
        LLM_GENERATION_DURATION.labels(provider_name).observe(generation_seconds)
    if isinstance(output_tokens, (int, float)):
        LLM_OUTPUT_TOKENS.labels(provider_name).observe(output_tokens)
    if tokens_per_second is not None:
        LLM_TOKENS_PER_SECOND.labels(provider_name).observe(tokens_per_second)

    measurement = {
        "provider": provider_name,
        "model": _model_name(provider),
        "mode": mode,
        "outcome": outcome,
        "call_index": call_index,
        "duration_seconds": elapsed,
        "ttft_seconds": ttft,
        "generation_seconds": generation_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
    }
    logger.info(
        "LLM call conversation_id=%s provider=%s model=%s mode=%s call=%d "
        "status=%s total_ms=%.1f ttft_ms=%s generation_ms=%s input_tokens=%s "
        "output_tokens=%s total_tokens=%s tps=%s",
        request_id or "-",
        measurement["provider"],
        measurement["model"],
        mode,
        call_index,
        outcome,
        elapsed * 1000,
        "-" if ttft is None else round(ttft * 1000, 1),
        "-" if generation_seconds is None else round(generation_seconds * 1000, 1),
        "unavailable" if input_tokens is None else input_tokens,
        "unavailable" if output_tokens is None else output_tokens,
        "unavailable" if total_tokens is None else total_tokens,
        "-" if tokens_per_second is None else round(tokens_per_second, 1),
    )
    return measurement


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
                    request_timeout=settings.llm_timeout_seconds,
                )
            )
        if settings.gemini_api_key:
            configured_models.append(
                ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    temperature=0,
                    api_key=settings.gemini_api_key,
                    timeout=settings.llm_timeout_seconds,
                )
            )
        if settings.openai_api_key:
            configured_models.append(
                ChatOpenAI(
                    model=settings.openai_model,
                    api_key=settings.openai_api_key,
                    request_timeout=settings.llm_timeout_seconds,
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
    for call_index, model in enumerate(models, 1):
        started = perf_counter()
        try:
            response = model.invoke(messages)
            _record_call(
                model, "invoke", "ok", started, request_id, call_index,
                response=response,
            )
            return response
        except Exception as exc:
            _record_call(model, "invoke", "error", started, request_id, call_index)
            provider = _provider_name(model)
            LLM_FAILURES.labels(provider).inc()
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


def stream_with_fallbacks(
    models, messages, writer, request_id: str = "", metrics_callback=None
):
    failures = []
    for call_index, model in enumerate(models, 1):
        try:
            response = _stream_model(
                model, messages, writer, request_id, call_index, metrics_callback
            )
            return response
        except Exception as exc:
            if getattr(exc, "output_started", False):
                raise RuntimeError(
                    f"Chat model {_provider_name(model)} streaming failed after "
                    "output began; refusing to retry to avoid duplicating a "
                    f"partial response: {type(exc).__name__}: {exc}"
                ) from exc
            provider = _provider_name(model)
            LLM_FAILURES.labels(provider).inc()
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


def _collect_stream(model, messages, writer):
    response = None
    first_text_at = None
    last_text_at = None
    try:
        for chunk in model.stream(messages):
            text = content_text(chunk.content)
            if text:
                now = perf_counter()
                if first_text_at is None:
                    first_text_at = now
                last_text_at = now
                writer({"type": "token", "text": text})
            response = chunk if response is None else response + chunk
    except Exception as exc:
        exc.first_text_at = first_text_at
        exc.last_text_at = last_text_at
        raise
    return response, first_text_at, last_text_at


def _record_stream_call(
    model,
    outcome,
    started,
    request_id,
    call_index,
    metrics_callback,
    first_text_at=None,
    last_text_at=None,
    response=None,
):
    ttft = first_text_at - started if first_text_at is not None else None
    generation = (
        last_text_at - first_text_at
        if first_text_at is not None and last_text_at is not None
        else None
    )
    measurement = _record_call(
        model,
        "stream",
        outcome,
        started,
        request_id,
        call_index,
        ttft=ttft,
        generation_seconds=generation,
        response=response,
    )
    if metrics_callback is not None:
        metrics_callback(measurement)
    return ttft


def _stream_model(
    model, messages, writer, request_id="", call_index=1, metrics_callback=None
) -> AIMessageChunk:
    started = perf_counter()
    try:
        response, first_text_at, last_text_at = _collect_stream(
            model, messages, writer
        )
    except Exception as exc:
        _record_stream_call(
            model,
            "error",
            started,
            request_id,
            call_index,
            metrics_callback,
            getattr(exc, "first_text_at", None),
            getattr(exc, "last_text_at", None),
        )
        if getattr(exc, "first_text_at", None) is not None:
            exc.output_started = True
        raise
    if response is None:
        _record_stream_call(
            model, "empty", started, request_id, call_index, metrics_callback
        )
        raise RuntimeError("The model returned no response chunks")
    _record_stream_call(
        model,
        "ok",
        started,
        request_id,
        call_index,
        metrics_callback,
        first_text_at,
        last_text_at,
        response,
    )
    return response
