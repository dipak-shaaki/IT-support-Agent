from prometheus_client import Counter, Histogram

# shortcut: metrics are per-process; use prometheus_client multiprocess mode
# (PROMETHEUS_MULTIPROC_DIR) if the API runs with multiple workers.
HTTP_REQUESTS = Counter(
    "it_support_http_requests_total",
    "HTTP requests handled.",
    ["method", "path", "status"],
)
HTTP_DURATION = Histogram(
    "it_support_http_request_duration_seconds",
    "HTTP request duration.",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
CHAT_REQUESTS = Counter(
    "it_support_chat_requests_total",
    "Chat requests by outcome and evidence source.",
    ["source", "status"],
)
INGEST_REQUESTS = Counter(
    "it_support_ingest_requests_total",
    "Document ingestion requests by outcome.",
    ["status"],
)
RATE_LIMITED = Counter(
    "it_support_rate_limited_total",
    "Requests rejected by the per-identity rate limiter.",
)
RETRIEVAL_DURATION = Histogram(
    "it_support_private_kb_retrieval_duration_seconds",
    "Elapsed time for private knowledge-base retrieval.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30),
)
LLM_FAILURES = Counter(
    "it_support_llm_failures_total",
    "Chat model provider calls that failed.",
    ["provider"],
)
LLM_CALLS = Counter(
    "it_support_llm_calls_total",
    "LLM calls by provider, transport mode, and outcome.",
    ["provider", "mode", "outcome"],
)
LLM_CALL_DURATION = Histogram(
    "it_support_llm_call_duration_seconds",
    "Elapsed time for an LLM call, including time to receive the full response.",
    ["provider", "mode"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)
LLM_TTFT = Histogram(
    "it_support_llm_time_to_first_token_seconds",
    "Time from starting a streamed LLM call to its first non-empty text chunk.",
    ["provider"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60),
)
LLM_GENERATION_DURATION = Histogram(
    "it_support_llm_generation_duration_seconds",
    "Time from the first to the last non-empty text chunk in a streamed LLM call.",
    ["provider"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60),
)
LLM_OUTPUT_TOKENS = Histogram(
    "it_support_llm_output_tokens",
    "Provider-reported output tokens per LLM call when available.",
    ["provider"],
    buckets=(1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
)
LLM_TOKENS_PER_SECOND = Histogram(
    "it_support_llm_output_tokens_per_second",
    "Provider-reported output tokens per second after the first streamed text chunk.",
    ["provider"],
    buckets=(1, 5, 10, 20, 40, 60, 100, 150, 250),
)
