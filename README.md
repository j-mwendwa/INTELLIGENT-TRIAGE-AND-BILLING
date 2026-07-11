# Intelligent Triage & Billing

> **AI-powered hierarchical multi-agent support platform** — routes customer queries to specialised Billing, Technical, and Compliance subgraph agents using LangGraph + Gemini + RAG.

---

## Architecture

```
HTTP Client
    │  POST /api/v1/chat  (or /chat/stream)
    ▼
FastAPI  ──► RequestIDMiddleware · TimingMiddleware · SecurityHeadersMiddleware · SlowAPI rate-limit
    │  Auth: X-API-Key header
    ▼
LangGraph Supervisor Graph
    │
    ├─► input_guardrail  (length · blocked-phrases · injection regex)
    │       ├─► [BLOCKED] → rejection_node → output_guard → END
    │       └─► [PASS]    → supervisor_node
    │
    ├─► supervisor_node  (Gemini classifies intent)
    │       ├─► billing    → billing_agent    (billing_search tool)
    │       ├─► technical  → technical_agent  (technical_search tool)
    │       ├─► compliance → compliance_agent (compliance_search tool)
    │       ├─► general    → general_agent    (all search tools)
    │       └─► escalate   → escalation_node
    │
    ├─► domain_agent  ◄──────────────────────────────┐
    │       └─► tool_calls? → tool_node ─────────────┘ (loop, max 8)
    │           no? → extract_answer
    │
    └─► output_guardrail → END → response to client
```

**Stack:** Python 3.11 · FastAPI · LangGraph · LlamaIndex · Gemini 2.5 Flash · Chroma/Qdrant

---

## Quick Start

### 1. Install

```bash
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY and ALLOWED_API_KEYS

make install
```

### 2. Add documents to knowledge bases

```bash
# Put .txt, .md, .pdf files in the relevant directories:
data/raw/billing/      ← billing policies, pricing, invoice guides
data/raw/technical/    ← API docs, troubleshooting guides, changelogs
data/raw/compliance/   ← GDPR docs, ToS, security policies

# Then ingest:
make ingest-billing
make ingest-technical
make ingest-compliance
```

### 3. Start the server

```bash
make serve
# → http://localhost:8000       (Chat UI)
# → http://localhost:8000/docs  (API docs)
```

### 4. Chat

Open `http://localhost:8000` in your browser, enter your API key in Settings ⚙️, and start chatting.

Or use the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Why was I charged $150 on invoice #INV-2024-001?", "thread_id": "thread-abc"}'
```

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Liveness probe |
| GET | `/version` | No | Version info |
| POST | `/api/v1/chat` | Yes | Synchronous chat |
| POST | `/api/v1/chat/stream` | Yes | Streaming SSE chat |
| DELETE | `/api/v1/threads/{id}` | Yes | Delete thread + entity memory |
| POST | `/api/v1/ingest` | Yes | Ingest from server path |
| POST | `/api/v1/ingest/upload` | Yes | Upload files then ingest |

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | Secrets (API keys, hosts) |
| `configs/config.yaml` | Runtime tuning (models, thresholds, MCP) |
| `prompts/system/` | Markdown system prompts (versioned) |

Key config knobs in `configs/config.yaml`:

```yaml
llm:
  default_model: gemini-2.5-flash   # change to gemini-2.5-pro for harder tasks

retrieval:
  top_k: 5
  similarity_cutoff: 0.7            # lower = more results, less precise
  temperature: 0.0

graph:
  max_iterations: 8                 # tool-call loop limit

supervisor:
  confidence_threshold: 0.6        # below this → general agent
```

---

## Developer Workflow

```bash
make install         # install + pre-commit hooks
make serve           # start dev server (hot reload)

# Ingestion
make ingest-billing
make ingest-technical
make ingest-compliance

# Testing
make test-unit       # fast, no external calls
make test-int        # requires GOOGLE_API_KEY

# Evaluation
make eval-rag        # retrieval quality
make eval-agent      # full agent routing accuracy + latency

# Context audit (see what the LLM sees)
python scripts/audit_context.py --message "Why was I charged $150?"

# Docker
make docker-up       # start Chroma
make docker-down
```

---

## Project Structure

```
src/
├── api/             FastAPI app, routes, auth, schemas
├── config.py        Pydantic settings + YAML config
├── core/            Context assembly, token counting, logging, prompts
├── graph/           LangGraph: state, nodes, edges, guardrails, graph
├── ingestion/       LlamaIndex document ingestion pipeline
├── memory/          Entity memory (JSON + Fernet) + conversation summary
├── retrieval/       LlamaIndex retriever wrapper
├── tools/           LangChain tools: billing/technical/compliance search + MCP
└── vectordb/        Chroma + Qdrant backends (factory pattern)

configs/config.yaml  Runtime tuning
prompts/             Versioned markdown prompt files
data/raw/            Source documents (billing/, technical/, compliance/)
web/index.html       Chat UI (served at /)
tests/unit/          Fast unit tests (no external calls)
tests/integration/   Integration tests (requires GOOGLE_API_KEY)
evals/               RAG and agent evaluation pipelines
scripts/             CLI tools: ingest.py, audit_context.py
```

---

## Security

| Layer | Mechanism |
|---|---|
| API auth | `X-API-Key` → SHA-256 audit log (key never logged) |
| Rate limiting | SlowAPI per-IP, default 30/min |
| Input guardrail | Length cap, blocked phrases, injection regex, leet-normalisation |
| Output guardrail | Empty-answer fallback, 16K char truncation |
| Path traversal | `os.walk(followlinks=False)` + allowlist root check |
| Memory encryption | Optional Fernet at rest (`MEMORY_ENCRYPTION_KEY`) |
| Security headers | X-Content-Type-Options, X-Frame-Options, etc. |
| Production guard | Startup fails if default API key used in production |

---

## License

MIT
