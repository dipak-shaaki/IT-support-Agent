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
LLM_FAILURES = Counter(
    "it_support_llm_failures_total",
    "Chat model provider calls that failed.",
    ["provider"],
)
