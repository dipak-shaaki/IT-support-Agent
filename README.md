# IT Support Agent

An internal IT-support assistant that answers from a private company knowledge base. Tavily public web search is reserved for live outage and service-status questions. The application includes a streaming chat UI, document ingestion, and an audit dashboard.

## Features

- Model-directed tool calling built with LangChain and LangGraph.
- A ReAct-style agent loop where the Groq model requests tools and LangGraph's `ToolNode` executes them.
- Private knowledge retrieval using Pinecone and local Hugging Face embeddings.
- Hybrid retrieval that fuses Pinecone vector results with a local BM25 index using reciprocal rank fusion (RRF).
- Private-KB-first tool availability and a bounded model/tool loop.
- Tavily web search for live outage and service-status questions only, after the private knowledge-base lookup.
- FastAPI endpoints for chat, token-streaming chat, document ingestion, health, and audit statistics.
- Redis-backed conversation memory with configurable expiration and a browser "New chat" control.
- Browser-based chat and operational dashboard.
- SQLite audit history for answer source, traces, and unanswered knowledge-base gaps.
- Locked Python dependencies managed with `uv`.

## How it works

```text
Question → Groq agent → tool call? ── no ──→ Direct response
                 │
                 yes
                 ↓
          LangGraph ToolNode
          ┌──────┴────────────┐
          ↓                   ↓
   Private KB search     Tavily web search
          └──────┬────────────┘
                 ↓
          Tool result to agent
                 ↓
        Grounded final response
```

The model requests tools using native tool calls; it does not merely receive a prompt asking it to imitate a call. A LangGraph `ToolNode` executes each requested function and returns its result to the model. The graph exposes only the private-KB tool for ordinary questions. Tavily is made available only for outage or live service-status requests, and only after the private-KB lookup.

Questions about live service status or outages are checked against the private KB first, then automatically sent to web search because internal runbooks cannot establish current vendor status. All other requests—including vendor setup, troubleshooting, and external organizations' policies—are answered from the private KB or identified as undocumented; they do not trigger public search. Employee-specific or confidential information is never sent to public search.

## Technology

| Area | Tools |
|---|---|
| API and application | FastAPI, Uvicorn |
| Agent workflow and tool calling | LangGraph `StateGraph` / `ToolNode`, LangChain tools |
| Language model | Groq |
| Private retrieval | Pinecone, Hugging Face sentence-transformers, BM25, reciprocal rank fusion |
| Web search | Tavily |
| Document ingestion | PyPDF, python-docx, LangChain loaders and text splitters |
| User interface | React, TypeScript, Vite |
| Audit storage | SQLite |
| Python environment and dependencies | uv |

## Project layout

```text
.
├── app/
│   ├── agent/                  # Agent policy, prompts, tools, graph, and state
│   │   ├── graph.py            # LangGraph assembly and ask() entry point
│   │   ├── model.py            # LLM providers and fallback behavior
│   │   ├── nodes.py            # Graph node implementations and routing
│   │   ├── policy.py           # Search policy classification
│   │   ├── prompts.py          # Agent system prompt
│   │   ├── state.py            # LangGraph state schema
│   │   └── tools.py            # Private-KB and public web tools
│   ├── api/                    # HTTP routes and request schemas
│   ├── core/                   # Settings and logging
│   ├── rag/                    # Retrieval, vector store, compatibility exports
│   ├── services/               # Document ingestion, conversation, and audit
│   └── main.py                 # FastAPI app and built frontend hosting
├── frontend/
│   ├── src/api/                # Typed calls to the FastAPI endpoints
│   ├── src/components/         # Shared React UI components
│   ├── src/pages/              # Chat and operations dashboard
│   ├── .env.example            # Public frontend configuration template
│   └── package.json            # Frontend dependencies and scripts
├── evals/
│   ├── cases.jsonl             # Labeled agent evaluation scenarios
│   └── run.py                  # Live evaluation runner
├── data/sample_kb/             # Sample IT documents to index
├── docs/                       # Project background documents
├── tests/                      # API, ingestion, audit, and tool-calling tests
├── uploads/                    # Uploaded documents
├── .env.example                # Environment variable template
├── Dockerfile
├── pyproject.toml              # Project metadata and dependencies
├── frontend/package-lock.json  # Reproducible frontend dependency lock
└── uv.lock                     # Reproducible dependency lock
```

Agent orchestration lives in `app/agent`; retrieval/indexing stays in `app/rag`,
and transport and persistence remain in `app/api` and `app/services`. The former
`app.rag.workflow` and `app.rag.state` imports are retained as compatibility
exports; new code should import from `app.agent`. The React app in `frontend/`
calls the existing FastAPI endpoints. FastAPI serves the production Vite build
from `frontend/dist`; local frontend development uses Vite's `/api` proxy.

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Node.js 20.19 or newer (or 22.12 or newer) and npm.
- API credentials for Groq, Tavily, and Pinecone.

## Set up

Create the environment and install the locked application and development dependencies:

```bash
uv sync --locked --group dev
```

Copy the example configuration and add your API credentials:

```bash
cp .env.example .env
```

Configure at least one of `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` in `.env`. When multiple keys are set, the agent tries Groq first, then Gemini, then OpenAI if a provider call fails. `TAVILY_API_KEY` and `PINECONE_API_KEY` are needed for web search and private retrieval. Redis must be available at `REDIS_URL` (defaults to `redis://localhost:6379/0`). Set `ADMIN_API_KEY` to a private value before using the dashboard or document-ingestion endpoint. The default value is for local development only.

Install and start Redis on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable --now redis-server
redis-cli ping
```

Redis should respond with `PONG`. Make sure `.env` uses
`REDIS_URL=redis://localhost:6379/0`. Run the backend in one terminal:

```bash
uv run --env-file .env uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Start the React development server in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173) for chat or
[http://127.0.0.1:5173/dashboard](http://127.0.0.1:5173/dashboard) for the
operations dashboard. Vite proxies `/api` requests to FastAPI on port 8080.
FastAPI's interactive API documentation remains at
[http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs). Port 8080 matches
`run.py` and the Docker image and avoids clashing with tools that probe the
common vLLM port 8000.

To build and serve the frontend through FastAPI locally:

```bash
cd frontend && npm ci && npm run build
cd ..
uv run --env-file .env uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080`. The Docker image builds the frontend and
serves it from FastAPI automatically.

## API overview

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check (also available as `/health` for probes) |
| `GET` | `/api/ready` | Readiness check for Redis and the keyword index |
| `GET` | `/metrics` | Prometheus metrics for HTTP, chat, ingestion, and rate limiting |
| `POST` | `/api/chat` | Ask a question and receive a complete response |
| `POST` | `/api/chat/stream` | Ask a question and receive server-sent events |
| `GET` | `/api/conversations/{conversation_id}` | Retrieve a conversation's recent chat transcript |
| `POST` | `/api/ingest` | Upload and index or replace a PDF, TXT, Markdown, or DOCX file |
| `GET` | `/api/ingest/jobs/{job_id}` | Check the status of a queued ingestion job |
| `GET` | `/api/stats` | Read audit statistics; requires the `x-admin-key` header |

Example request:

```bash
curl -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I connect to the company VPN?"}'
```

To index a document, send a supported file to `/api/ingest` with the `x-admin-key` header set to `ADMIN_API_KEY`. The file is parsed, chunked, embedded, and stored in Pinecone; the local BM25 corpus is updated too. Uploading a revised file with the same filename replaces its previous indexed chunks. The statistics endpoint uses the same header. Uploaded files are deleted after they are indexed.

To index a whole directory without the UI, run the bulk helper from the repository root. Use `--dry-run` first to parse and count chunks without calling Pinecone:

```bash
uv run --env-file .env python -m app.bulk_ingest data/sample_kb --dry-run
uv run --env-file .env python -m app.bulk_ingest data/sample_kb
```

Markdown files that use the `## IT-xxxxx` Q&A format are parsed as one chunk per record, so a question and its answer stay together. Record metadata (`record_id`, `question`, `category`, `record_number`, `document_type`, `language`) is stored in Pinecone and the BM25 corpus, and private-KB citations use the record ID. Other Markdown, PDF, text, and Word files use the generic section splitter.

With `INGEST_MODE=queue`, the endpoint stores the upload, returns HTTP 202 with a `job_id`, and a worker processes the file in the background; poll `/api/ingest/jobs/{job_id}` for `queued`, `running`, `done`, or `error`. A running job carries a lease; if the worker dies, the job is requeued after `INGEST_LEASE_SECONDS` and marked failed after `INGEST_MAX_ATTEMPTS`. Docker Compose sets this mode and runs the worker as a separate service. The worker logs and retries with backoff while Redis is unavailable. The app and worker share the `data` and `uploads` volumes, which is required for the worker's BM25 updates to reach the API process.

```bash
curl -X POST http://127.0.0.1:8080/api/ingest \
  -H 'x-admin-key: YOUR_ADMIN_API_KEY' \
  -F 'file=@data/sample_kb/service_desk_runbook.md'
```

Chat requests may include a `conversation_id` UUID. If omitted, the API creates one and returns it in the response (or the stream's `done` event). Reuse that ID on subsequent requests to continue the conversation. The web UI stores the ID in browser local storage and restores the transcript; use **New chat** to start a separate conversation. The server retains up to 20 messages (10 exchanges) and expires inactive conversations after 24 hours by default.

The streaming endpoint sends `token` events as the model generates its final answer, `trace` events as agent/tool steps complete, and a final `done` event with the complete answer and citations. Failures are sent as `error` events.

Chat questions are trimmed, must not be blank, and are limited to 400 tokens using the `cl100k_base` tokenizer, plus a 12,000-character hard limit to bound request processing. Token counts are an estimate for providers with different tokenizers. The API validates request and response shapes; oversized questions return HTTP 422 (or an SSE request error before streaming begins).

Backend logs include the submitted question's character and `cl100k_base` token counts. Each model call logs provider-reported input, output, and total tokens, correlated by conversation ID; these counts include separate model/tool-loop calls and should be summed for a turn. If a provider omits usage metadata, its model-call token counts are logged as `unavailable`. Question contents and model prompts are not written to these token-usage log entries.

## Configuration

| Variable | Purpose |
|---|---|
| `APP_NAME` | Application name shown in the UI and health response |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq credentials and model; first provider in the fallback order |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini credentials and model; defaults to `gemini-3.8-flash` |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI credentials and model; defaults to `gpt-5.4-mini` |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_AGENT_ID`, `LANGSMITH_AGENT_ENVIRONMENT` | Enable LangSmith tracing and route runs to an agent environment |
| `TAVILY_API_KEY` | Tavily web-search credentials |
| `PINECONE_API_KEY` | Pinecone credentials |
| `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE` | Private vector index and namespace |
| `EMBEDDING_MODEL` | Local FastEmbed/ONNX embedding model; defaults to `sentence-transformers/all-MiniLM-L6-v2` |
| `TOP_K` | Number of hybrid private knowledge-base results to return; defaults to 5 |
| `REDIS_URL` | Redis connection URL for conversation history |
| `CONVERSATION_TTL_SECONDS` | Inactivity lifetime for a conversation; defaults to 86400 seconds |
| `ADMIN_API_KEY` | Shared key for the audit statistics and ingestion endpoints |
| `MAX_UPLOAD_BYTES` | Maximum accepted upload size; defaults to 25 MB |
| `AUTH_MODE` | `none` for local use, `proxy` to trust an SSO identity header; required in production |
| `AUTH_USER_HEADER` | Identity header set by the trusted reverse proxy; defaults to `X-Auth-Request-User` |
| `AUTH_ROLES_HEADER` | Role header set by the trusted reverse proxy; defaults to `X-Auth-Request-Roles` |
| `AUTH_ADMIN_ROLES` | Comma-separated proxy roles that grant admin access; empty disables role-based admin |
| `RATE_LIMIT_PER_MINUTE` | Per-identity chat request limit in a one-minute window; defaults to 30 |
| `LLM_TIMEOUT_SECONDS` | Per-request provider timeout; defaults to 60 seconds |
| `INGEST_MODE` | `inline` (default) or `queue`; queue mode requires a running worker |
| `INGEST_JOB_TTL_SECONDS` | How long queued job status is retained in Redis; defaults to 86400 seconds |
| `INGEST_LEASE_SECONDS` | How long a worker may hold a running job before it is requeued; defaults to 300 seconds |
| `INGEST_MAX_ATTEMPTS` | How many attempts a stalled job gets before it is marked failed; defaults to 3 |
| `ALLOW_INGESTION` | Upload endpoints are enabled outside production and disabled in production; set `true` to opt in |

At least one model API key is required. The default models are `openai/gpt-oss-120b` for Groq, `gemini-3.8-flash` for Gemini, and `gpt-5.4-mini` for OpenAI. Fallbacks apply to request failures before a streamed response starts; a stream that fails after sending tokens is not retried to avoid duplicating output.

The optional `frontend/.env` file may define `VITE_API_BASE_URL` when the UI is
hosted separately from FastAPI. Leave it empty for the same-origin deployment
and Vite's local API proxy. Any `VITE_*` setting is embedded in browser assets
and is public; never put API keys or other secrets in frontend environment
variables.

For a small deployment such as a 512 MB Render instance, the embedding model runs through FastEmbed's ONNX runtime instead of torch. Ingestion is disabled when `APP_ENV=production`; index the corpus locally with `app.bulk_ingest` and deploy retrieval only. Set `ALLOW_INGESTION=true` only when you deliberately want uploads handled by the deployed app.

For an agent-based LangSmith workspace, create the agent and environment first, then set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_AGENT_ID`, and `LANGSMITH_AGENT_ENVIRONMENT` in `.env`. Use the agent's identifier (not its display name) and an existing environment such as `development`. Start commands with `uv run --env-file .env ...`. Traces may contain user prompts, retrieved documents, and model answers; enable tracing only with data approved for the configured LangSmith workspace.

Additional paths such as `AUDIT_DB_PATH`, `BM25_INDEX_PATH`, `UPLOAD_DIR`, and `SAMPLE_KB_DIR` have defaults in `app/core/config.py`.

The BM25 corpus is written to `data/bm25_corpus.jsonl` by ingestion. Documents indexed before hybrid retrieval was enabled must be ingested again to appear in keyword results. If the corpus is missing or empty, retrieval logs a warning and falls back to vector-only. In queue mode the corpus file must be on storage shared by the app and worker; the Compose setup uses a named volume, while a multi-host deployment needs a shared filesystem. The corpus is gitignored, so the Docker image regenerates it from `data/sample_kb` at build time with `python -m app.build_bm25`; it does not download documents or call an embedding model.

## Tests

Run the test suite with:

```bash
uv run --locked --group dev pytest
```

The tests cover API validation and responses, conversation-memory persistence and follow-ups, admin-key protection, document loading and chunking, tool calling, evaluation scoring, and audit statistics. They use local test data and mocks; they do not require live Redis, LLM, search, or vector-database calls.

## Continuous integration

GitHub Actions runs the test suite on pushes to `main`, pull requests targeting `main`, and manual workflow dispatch. CI checks the committed `uv.lock`, installs the locked development dependencies, and runs tests on Python 3.11 and 3.14. It uses temporary audit and upload paths and does not require production API credentials or deploy the application.

## Agent evaluations

The test suite checks implementation behavior; it does not, by itself, measure whether live model responses are useful. `evals/cases.jsonl` is a labeled live-agent test set covering common internal IT questions, direct greetings, current service status, an external-company policy question, and requests for unavailable personal information. Cases define the expected answer source, tool choices, citations, supporting evidence, and relevant knowledge-base source files.

Put representative, reviewed user questions in this file and label which source files should answer each retrieval case. To run the live evaluation, configure Groq, Pinecone, and Tavily credentials and ensure the Pinecone namespace contains the relevant company documents. Documents can be indexed through the UI's **Add documents** action. Redis is not needed for evaluation:

```bash
uv run --env-file .env --locked --group dev python -m evals.run
```

The checked-in benchmark currently has 45 scenarios, including 41 with relevant-source labels. The runner calls the configured Groq model and Pinecone index (and Tavily for current-status cases only). Save a report and repeat each case to collect latency samples:

```bash
uv run --env-file .env --locked --group dev python -m evals.run \
  --repeats 3 \
  --output evals/reports/run.json
```

This makes 135 sequential agent requests; use one repeat first to check credentials/indexing. Keep the report with the code/dataset revision and compare runs only when model, prompt, corpus, and retrieval settings are unchanged. The output includes a dataset SHA-256, model/retrieval metadata, per-case results, and aggregate metrics:

- **Source accuracy**: fraction of cases routed to the expected answer source.
- **Tool-selection accuracy**: fraction of labeled cases whose exact tool sequence matches the expected sequence.
- **Retrieval precision@5, recall@5, hit-rate@5, and MRR@5**: source-file-level metrics for cases with `expected_relevant_sources` labels. Chunks from a file count as one ranked source. Precision@5 uses a fixed denominator of five; recall@5 counts distinct relevant source files found in the first five unique sources. The report also includes the previous all-retrieved-source precision, recall, and MRR.
- **Latency p95/p99**: per-question elapsed time around `ask()`, including model and tool calls, but excluding app/UI streaming. Percentiles use the nearest-rank method and include failed calls. Repeats include cold-start effects in the first run; inspect per-case timings and use the same setup when comparing results.
- **Pass rate**: fraction of cases that meet all their specified checks, including answer terms and citation requirements.

These are checks over a small, hand-labeled set; answer-term checks are not a semantic faithfulness score or a substitute for human review. The current questions were derived from the sample handbook/runbook and should be reviewed against the intended corpus. Live model and web-search behavior can vary. The benchmark does not calculate Ragas faithfulness or answer relevancy. Keep this live evaluation manual; CI runs offline unit tests and does not call external providers.

## Docker (optional)

If you prefer containers, start the app and Redis with Compose after creating `.env`:

```bash
docker compose up --build
```

Compose starts Redis with persistent storage and configures the app to connect to it. Do not expose Redis publicly; configure access controls and TLS when connecting to a managed Redis service.

## Security notes

- Never commit `.env` or paste API keys into source files.
- Replace the development `ADMIN_API_KEY` with a unique secret before use.
- Conversation IDs grant access to their chat history; treat them as private and protect Redis data and backups according to your organization's retention policy.
- Protect the application and its audit data with your organization's authentication, authorization, network, and retention controls before exposing it beyond a trusted environment.
- In production, terminate authentication at a trusted reverse proxy and set `AUTH_MODE=proxy`. The proxy must overwrite `AUTH_USER_HEADER`; never accept that header directly from clients. Conversation history is scoped to that identity in Redis.
- To replace the shared admin key with proxy-managed roles, set `AUTH_ADMIN_ROLES` to one or more comma-separated roles and have the proxy overwrite `AUTH_ROLES_HEADER`. The proxy must strip both headers from client requests. An admin API key is still accepted for local and break-glass access.
- `/metrics` is unauthenticated. Expose it only on an internal network or protect it at the reverse proxy; do not publish it with the public chat UI.
- Only upload documents that you are authorized to process and index.
