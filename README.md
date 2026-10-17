# WA-Agent

**A WhatsApp-style conversational AI agent with a production-shaped memory pipeline.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-black.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Postgres](https://img.shields.io/badge/store-Postgres%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

WA-Agent is a customer-support agent for WhatsApp-style messaging. Each incoming
message is handled by a LangGraph state machine that **plans → retrieves memory →
calls tools → responds**, and hands off to a human when confidence is low. It
maintains **short-term session memory** (Redis) and **long-term semantic memory**
(Postgres + pgvector), and ships with an evaluation harness that measures the
impact of memory on answer accuracy.

The system runs **fully offline by default** — a deterministic mock LLM, local
embeddings, and an automatic SQLite fallback — so it can be cloned, run, and
evaluated with no API key. Switching to a real model requires only environment
variables.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [The Memory Pipeline](#the-memory-pipeline)
- [Evaluation](#evaluation)
- [Quickstart](#quickstart)
- [Running with a Real Model](#running-with-a-real-model)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Tools](#tools)
- [MCP Server](#mcp-server)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Testing](#testing)
- [What Is Real vs. Stubbed](#what-is-real-vs-stubbed)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Agentic orchestration** — a LangGraph state graph with typed state, a
  checkpointer, and an `interrupt()`-based human handoff.
- **Two-tier memory** — Redis-backed short-term session buffer with rolling
  summary compaction, plus Postgres/pgvector long-term semantic memory.
- **Write policy** — a pure, unit-testable function decides which facts are
  worth persisting (durable, user-specific, non-PII) before they are stored.
- **Typed tool system** — a read-only registry of Pydantic-validated tools, also
  exposed over the Model Context Protocol (MCP).
- **Async-first webhook** — a FastAPI endpoint with HMAC signature verification,
  rate limiting, and queue-backed processing off the request path.
- **Offline-first** — mock LLM, deterministic embeddings, and automatic
  SQLite/dict fallbacks so the pipeline never hard-fails.
- **Evaluation harness** — golden conversations scored with memory ON vs. OFF,
  reporting both accuracy and token-cost deltas.

---

## Architecture

```
                     WhatsApp Cloud API  /  local simulator
                                |
                     +----------v-----------+     signature check,
                     |   WEBHOOK (FastAPI)  |     rate limit,
                     |  POST /webhook/...   |     enqueue -> 202
                     +----------+-----------+
                                |  Redis queue
                     +----------v-----------+
                     |       WORKER         |     agent runs off
                     +----------+-----------+     the request path
                                v
                     +----------------------+
                     |   LANGGRAPH AGENT    |
                     |                      |
                     |   planner            |  classify intent + select tools
                     |   retrieve           |  load summary + top-k memories
                     |   tools              |  run read-only lookups
                     |   responder          |  compose reply + confidence
                     |   human_handoff      |  interrupt() for a human
                     +----------+-----------+
             +------------------+------------------+
             v                  v                   v
      Redis sessions     pgvector memories    Postgres audit
      (turns+summary)    (semantic recall)    (messages, traces)
```

The app auto-detects Postgres and Redis at startup and falls back to SQLite and
an in-process dictionary when they are unavailable, so the same code runs in any
environment.

---

## The Memory Pipeline

The memory layer is the core of the project.

| Tier | Backend | Responsibility |
|------|---------|----------------|
| **Short-term** | Redis | Stores the last *N* turns per user. When the buffer exceeds a threshold, older turns are folded into a **rolling summary** (`compact`) so context never grows unbounded. |
| **Long-term** | Postgres + pgvector | A **write policy** extracts durable, user-specific, non-PII facts (`preference`, `fact`, `event`, `sentiment`). Each new message is embedded, and the top-*k* relevant memories are retrieved by cosine distance and injected into the responder prompt. |

- **PII guard** — phone numbers, emails, and card patterns are stripped before
  anything is persisted.
- **Embeddings** — a deterministic hashing embedder by default (stable across
  runs, so evaluations are reproducible); a real embedder is opt-in via
  `WA_EMBEDDER`.
- **Decay** — memories track an access count and can be archived when unused.

---

## Evaluation

`eval/golden_conversations.jsonl` defines **12 multi-turn conversations** across
seven categories. The harness (`eval/run_eval.py`) runs each conversation twice —
once with memory ON and once with it OFF — and reports accuracy and token cost.

```
                      Memory ON     Memory OFF
Overall                12 / 12          8 / 12
Accuracy                100.0%          66.7%
Tokens                   1065             960

By category              ON      OFF
  memory_recall          3/3     0/3
  pii                    1/1     0/1
  tool_use               2/2     2/2
  handoff                2/2     2/2
  faq                    2/2     2/2
  billing                1/1     1/1
  smalltalk              1/1     1/1

Accuracy delta: +33.3%   Token cost: +105
```

The `memory_recall` and `pii` rows are the evidence: identical conversations fail
without long-term memory and pass with it. Full results are written to
`eval/results/latest.json`.

---

## Quickstart

Requires Python 3.11. No Docker, no API key, and no network needed.

```bash
pip install -r requirements.txt

python run.py --demo            # scripted multi-turn demo (memory + handoff)
python run.py                   # interactive REPL
python eval/run_eval.py         # memory ON vs. OFF scorecard
python -m pytest tests -q       # 29 tests, fully offline
```

To run the HTTP service:

```bash
uvicorn waagent.api:app --port 8000
```

### Full stack (Postgres + pgvector + Redis + worker)

```bash
docker compose up --build
```

The schema in `db/migrations/0001_init.sql` is applied automatically. `api` and
`worker` share a single image.

**Sample session** (`python run.py --demo`):

```
you : hi
bot : Hi! How can I help you today? I can track orders, explain refunds, or answer FAQs.

you : I prefer Hindi and WhatsApp for updates
bot : Got it - I'll keep that in mind for our conversation.        [wrote 1 memory]

you : which language should you reply in for updates?
bot : (I remember: I prefer Hindi and WhatsApp for updates.)       [recalled 1]

you : where is my order 1234
bot : Your Wireless Earbuds is packed (ETA: 2 days).               [tool: get_order_status]

you : I want to talk to a human
bot : I'm connecting you to a human agent who can take it from here.
```

---

## Running with a Real Model

The agent talks to any OpenAI-compatible `/chat/completions` endpoint. Set the
following in `.env`:

```bash
WA_LLM_PROVIDER=groq                      # or: openai
WA_GROQ_API_KEY=your_key_here
WA_LLM_MODEL=openai/gpt-oss-120b          # any model available on your account

# Optional: real embeddings instead of the deterministic mock
WA_EMBEDDER=openai
WA_OPENAI_API_KEY=your_key_here
```

No code changes are required. The prompts in `waagent/prompts/` drive planning,
response, compaction, and memory classification. Verified on Groq with
`openai/gpt-oss-120b`.

---

## Configuration

All settings use the `WA_` prefix and are read from the environment or a local
`.env` file (see `.env.example`).

| Variable | Default | Purpose |
|----------|---------|---------|
| `WA_LLM_PROVIDER` | `mock` | `mock` \| `groq` \| `openai` |
| `WA_LLM_MODEL` | `openai/gpt-oss-120b` | Model name for the chosen provider |
| `WA_EMBEDDER` | `mock` | `mock` or `openai` |
| `WA_DATABASE_URL` | local Postgres | SQLAlchemy database URL |
| `WA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `WA_SESSION_MAX_TURNS` | `8` | Turn buffer size before compaction |
| `WA_SUMMARY_TRIGGER_TURNS` | `6` | Turns before the summary is rewritten |
| `WA_MEMORY_TOP_K` | `5` | Memories retrieved per turn |
| `WA_MIN_REPLY_CONFIDENCE` | `0.55` | Below this, the agent hands off |
| `WA_RATE_LIMIT_PER_MINUTE` | `20` | Per-user request limit |
| `WA_APP_SECRET` | *(empty)* | Meta app secret for webhook HMAC checks |

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Synchronous simulator endpoint |
| `POST` | `/chat/async` | Enqueued; poll `GET /jobs/{id}` for the result |
| `POST` | `/webhook/whatsapp` | Meta-shaped webhook (async) |
| `GET`  | `/webhook/whatsapp` | Webhook verification (`hub.challenge`) |
| `GET`  | `/healthz` | Health check |
| `GET`  | `/metrics` | Request counts, latency, handoff and memory-hit rates |

To connect the real WhatsApp Cloud API, set `WA_APP_SECRET`, point the Meta
webhook at `/webhook/whatsapp`, and complete the verification handshake. The
payload shape, HMAC `X-Hub-Signature-256` check, and async enqueue already match
the Cloud API; no code changes are required.

---

## Tools

All tools are typed, read-only, and validated with Pydantic input models.

| Tool | Description |
|------|-------------|
| `search_faq` | Keyword search over a seeded FAQ (stop-word filtered, overlap-gated) |
| `get_order_status` | Looks up an order by ID |
| `check_refund_policy` | Returns the refund policy |
| `get_session_info` | Returns current session details |
| `escalate_to_human` | Transfers the conversation to a human agent |

---

## MCP Server

The same tool registry is exposed as a Model Context Protocol server, so any MCP
client (Claude Desktop, Cursor) can drive the identical tools:

```
mcp_server/server.py   # FastMCP: search_faq, get_order_status,
                       # check_refund_policy, get_session_info, escalate_to_human
```

---

## Project Structure

```
waagent/
├── waagent/                  # Core package
│   ├── agent.py              #   LangGraph graph: planner -> retrieve -> tools -> responder -> handoff
│   ├── memory.py             #   Redis sessions, summary compaction, pgvector semantic memory
│   ├── llm.py                #   MockLLM + OpenAI-compatible client (Groq / OpenAI)
│   ├── embeddings.py         #   Deterministic mock + real embedder
│   ├── tools.py              #   Typed read-only tool registry
│   ├── api.py                #   FastAPI webhook, /chat, /metrics, /healthz
│   ├── queue.py, worker.py   #   Async job processing
│   ├── ratelimit.py          #   Redis token-bucket rate limiter
│   ├── schemas.py            #   Pydantic contracts
│   └── prompts/              #   planner.md, responder.md, compactor.md, memory_classifier.md
├── db/
│   ├── database.py           #   SQLAlchemy engine + session helpers
│   └── migrations/           #   0001_init.sql (Postgres/pgvector schema)
├── mcp_server/
│   └── server.py             #   FastMCP server exposing the 5 tools
├── eval/
│   ├── golden_conversations.jsonl
│   ├── run_eval.py           #   Memory ON vs. OFF scorecard
│   └── results/              #   latest.json
├── tests/                    # 29 tests: memory, agent, api, queue, ratelimit, mcp
├── run.py                    # Local chat simulator (REPL + --demo)
├── docker-compose.yml        # pgvector + redis + api + worker
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Agent orchestration | LangGraph (checkpointer + interrupts) |
| LLM | Mock (default) · Groq / OpenAI (opt-in) |
| Embeddings | Deterministic hashing (default) + OpenAI-compatible |
| API | FastAPI + Uvicorn |
| Datastores | Postgres 16 + pgvector; SQLite fallback |
| Cache / Queue | Redis 7; in-process fallback |
| Tool protocol | Model Context Protocol (mcp SDK, stdio) |
| Testing | pytest |

---

## Testing

```bash
python -m pytest tests -q      # 29 passed
```

The suite covers memory (compaction, write policy, PII stripping, retrieval),
the end-to-end agent loop, the API, the queue/worker, the rate limiter, and the
MCP server. All tests run offline against the mock LLM.

---

## What Is Real vs. Stubbed

**Real.** The LangGraph agent and interrupt handoff; the Redis/pgvector memory
pipeline and write policy; the FastAPI webhook with HMAC signature verification
and rate limiting; the async queue and worker; the evaluation harness.

**Stubbed.** The order, FAQ, and refund data in `waagent/tools.py` are realistic
shapes with no external system behind them. The human-handoff destination is a
placeholder.

**Not included.** A live Meta number (the webhook is simulator-first but
Meta-shaped), multi-tenant authentication, and cloud deployment tooling.

The Postgres/pgvector + Redis path is fully written and `docker compose config`
validates, but it has not been exercised end-to-end in this environment (Docker
was unavailable). All verified results were produced on the SQLite/dict fallback.

---

## Roadmap

- [ ] Shared Postgres/Redis checkpointer and distributed rate limiter.
- [ ] Run the Postgres/pgvector + Redis path end-to-end; add an ANN index.
- [ ] Real-LLM hardening: timeouts, retries, and guardrails.
- [ ] A proper queue (arq/Celery) with retries and a dead-letter queue.
- [ ] CI with ruff/mypy and Alembic migrations.
- [ ] Webhook idempotency.
- [ ] Deployment sketch (ECS + RDS + ElastiCache).

---

## License

Released under the MIT License. See `LICENSE` for details.
