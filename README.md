# IT Support Agent

An internal IT-support assistant that answers from a private company knowledge base first and uses public web search only when it is needed. The application includes a streaming chat UI, document ingestion, and an audit dashboard.

## Features

- Model-directed tool calling built with LangChain and LangGraph.
- A ReAct-style agent loop where the Groq model requests tools and LangGraph's `ToolNode` executes them.
- Private knowledge retrieval using Pinecone and local Hugging Face embeddings.
- Private-KB-first tool availability and a bounded model/tool loop.
- Tavily web search for current service status and relevant public vendor information when private guidance is insufficient.
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

The model requests tools using native tool calls; it does not merely receive a prompt asking it to imitate a call. A LangGraph `ToolNode` executes each requested function and returns its result to the model. The graph exposes only the private-KB tool initially; Tavily becomes available after a private search has run. For ordinary IT questions, the assistant uses web search only if the private results do not sufficiently answer the question.

Questions about live service status or outages are checked against the private KB first, then automatically sent to web search because internal runbooks cannot establish current vendor status. If a user explicitly asks about a named organization's public IT, employee, or workplace policy, the assistant also checks the private KB first and, if needed, searches for that organization's policy on its official website. It does not substitute another organization's policy or present public information as the user's employer's policy. A company mention alone does not trigger a web search, and employee-specific or confidential information is never sent to public search.

## Technology

| Area | Tools |
|---|---|
| API and application | FastAPI, Uvicorn |
| Agent workflow and tool calling | LangGraph `StateGraph` / `ToolNode`, LangChain tools |
| Language model | Groq |
| Private retrieval | Pinecone, Hugging Face sentence-transformers |
| Web search | Tavily |
| Document ingestion | PyPDF, python-docx, LangChain loaders and text splitters |
| User interface | HTML, CSS, JavaScript |
| Audit storage | SQLite |
| Python environment and dependencies | uv |

## Project layout

```text
.
├── app/
│   ├── api/routes.py           # HTTP API routes
│   ├── core/                   # Settings and logging
│   ├── rag/                    # Agent state, workflow, and vector store
│   ├── services/               # Document ingestion and audit storage
│   └── main.py                 # FastAPI app and web pages
├── evals/
│   ├── cases.jsonl             # Labeled agent evaluation scenarios
│   └── run.py                  # Live evaluation runner
├── data/sample_kb/             # Sample IT documents to index
├── docs/                       # Project background documents
├── static/                     # Browser JavaScript and CSS
├── templates/                  # Chat and dashboard pages
├── tests/                      # API, ingestion, audit, and tool-calling tests
├── uploads/                    # Uploaded documents
├── .env.example                # Environment variable template
├── Dockerfile
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Reproducible dependency lock
```

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
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

Set `GROQ_API_KEY`, `TAVILY_API_KEY`, and `PINECONE_API_KEY` in `.env`. Redis must be available at `REDIS_URL` (defaults to `redis://localhost:6379/0`). Set `ADMIN_API_KEY` to a private value before using the dashboard or document-ingestion endpoint. The default value is for local development only.

Install and start Redis on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable --now redis-server
redis-cli ping
```

Redis should respond with `PONG`. Make sure `.env` uses `REDIS_URL=redis://localhost:6379/0`, then run the app:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) for chat or [http://127.0.0.1:8080/dashboard](http://127.0.0.1:8080/dashboard) for the audit dashboard. FastAPI's interactive API documentation is at [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs).

## API overview

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat` | Ask a question and receive a complete response |
| `POST` | `/api/chat/stream` | Ask a question and receive server-sent events |
| `GET` | `/api/conversations/{conversation_id}` | Retrieve a conversation's recent chat transcript |
| `POST` | `/api/ingest` | Upload and index a PDF, TXT, Markdown, or DOCX file |
| `GET` | `/api/stats` | Read audit statistics; requires the `x-admin-key` header |

Example request:

```bash
curl -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I connect to the company VPN?"}'
```

To index a document, send a supported file to `/api/ingest` with the `x-admin-key` header set to `ADMIN_API_KEY`. The statistics endpoint uses the same header.

Chat requests may include a `conversation_id` UUID. If omitted, the API creates one and returns it in the response (or the stream's `done` event). Reuse that ID on subsequent requests to continue the conversation. The web UI stores the ID in browser local storage and restores the transcript; use **New chat** to start a separate conversation. The server retains up to 20 messages (10 exchanges) and expires inactive conversations after 24 hours by default.

The streaming endpoint sends `token` events as the model generates its final answer, `trace` events as agent/tool steps complete, and a final `done` event with the complete answer and citations. Failures are sent as `error` events.

## Configuration

| Variable | Purpose |
|---|---|
| `APP_NAME` | Application name shown in the UI and health response |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq credentials and model |
| `TAVILY_API_KEY` | Tavily web-search credentials |
| `PINECONE_API_KEY` | Pinecone credentials |
| `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE` | Private vector index and namespace |
| `EMBEDDING_MODEL` | Local sentence-transformers embedding model |
| `TOP_K` | Number of private knowledge-base results to retrieve |
| `REDIS_URL` | Redis connection URL for conversation history |
| `CONVERSATION_TTL_SECONDS` | Inactivity lifetime for a conversation; defaults to 86400 seconds |
| `ADMIN_API_KEY` | Shared key for the audit statistics and ingestion endpoints |

Additional paths such as `AUDIT_DB_PATH`, `UPLOAD_DIR`, and `SAMPLE_KB_DIR` have defaults in `app/core/config.py`.

## Tests

Run the test suite with:

```bash
uv run --locked --group dev pytest
```

The tests cover API validation and responses, conversation-memory persistence and follow-ups, admin-key protection, document loading and chunking, tool calling, evaluation scoring, and audit statistics. They use local test data and mocks; they do not require live Redis, LLM, search, or vector-database calls.

## Continuous integration

GitHub Actions runs the test suite on pushes to `main`, pull requests targeting `main`, and manual workflow dispatch. CI checks the committed `uv.lock`, installs the locked development dependencies, and runs tests on Python 3.11 and 3.14. It uses temporary audit and upload paths and does not require production API credentials or deploy the application.

## Agent evaluations

The test suite checks implementation behavior; it does not, by itself, measure whether live model responses are useful. `evals/cases.jsonl` is a small labeled evaluation set covering private-KB answers, direct greetings, web fallback, and abstention for unavailable personal information.

After setting `.env` and indexing the sample knowledge base, run the live evaluation. Redis is not needed for this command:

```bash
uv run python -m evals.run
```

The runner calls the configured Groq model and Pinecone index (and Tavily for the current-status case). It prints per-case source, tool-order, citation, retrieved-evidence, and answer-term checks plus an aggregate pass rate. It exits with status 1 if any case fails or a provider call errors. Web-search cases depend on live results, so scores can vary; use these checks as a regression signal, not as a claim of human-verified answer quality. Extend the labeled cases with representative questions and reviewed expected behaviors as the knowledge base changes.

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
- Only upload documents that you are authorized to process and index.
