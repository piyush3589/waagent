<div align="center">

# WA-Agent

### A WhatsApp-style conversational AI that actually remembers

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-state%20graph-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-webhook-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Postgres](https://img.shields.io/badge/Postgres-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/Tests-29%20passed-brightgreen?style=flat-square)](#-verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

*Plan &rarr; Retrieve memory &rarr; Call tools &rarr; Respond &rarr; Hand off to human — with short-term and long-term memory that measurably makes it smarter.*

<br>

</div>

---

## What is this?

A **production-shaped customer-support agent** for WhatsApp-style messaging. It answers questions, looks up data, **remembers users across conversations**, and hands off to a human when it should — with an eval scorecard that *proves* the memory layer works.

> Runs **100% offline** by default (mock LLM + deterministic embeddings + SQLite fallback). Flip two env vars to use a real model.

---

## See it in action

```console
$ python run.py --demo

you : hi
bot : Hi! How can I help you today? I can track orders, explain refunds, or answer FAQs.

you : I prefer Hindi and WhatsApp for updates
bot : Got it — I'll keep that in mind for our conversation.        💾 wrote 1 memory

you : which language should you reply in for updates?
bot : (I remember: I prefer Hindi and WhatsApp for updates.)       🧠 recalled 1

you : where is my order 1234
bot : Your Wireless Earbuds is packed (ETA: 2 days).               🔧 get_order_status

you : I want to talk to a human
bot : I'm connecting you to a human agent who can take it from here.
```

The bot **remembers the language preference** from two turns ago and **uses a tool** to look up the order — all offline, no API key.

---

## How it works

```
                      WhatsApp Cloud API  /  local simulator
                                 │
                      ┌──────────▼───────────┐
                      │   WEBHOOK (FastAPI)   │   HMAC signature check,
                      │  POST /webhook/...    │   rate limit, enqueue → 202
                      └──────────┬───────────┘
                                 │  Redis queue
                      ┌──────────▼───────────┐
                      │      WORKER           │   agent runs off the
                      └──────────┬───────────┘   request path
                                 ▼
                      ┌──────────────────────┐
                      │   LANGGRAPH AGENT     │
                      │                       │
                      │  🧠 planner           │  classify intent + pick tools
                      │  📚 retrieve          │  pull summary + top-k memories
                      │  🔧 tools             │  execute read-only lookups
                      │  💬 responder         │  compose reply + confidence
                      │  🙋 handoff           │  interrupt() for human
                      └──────────┬───────────┘
                ┌────────────────┼─────────────────┐
                ▼                ▼                 ▼
          Redis sessions    pgvector memories   Postgres audit
          (turns+summary)   (facts, retrieval)  (messages, traces)
```

Every message follows the same loop: **think &rarr; remember &rarr; act &rarr; reply**. If confidence is low or the user asks for a human, the graph **interrupts** and pauses cleanly.

---

## The memory pipeline (the whole point)

Most chatbots forget everything after a session ends. This one doesn't.

| Layer | Store | What it does |
|-------|-------|-------------|
| **Short-term** | Redis | Keeps the last N turns. When the buffer grows too long, older turns are **compacted into a rolling summary** so context never overflows. |
| **Long-term** | Postgres + pgvector | A **write-policy** extracts durable, user-specific, non-PII facts (`preference`, `fact`, `event`, `sentiment`). Each new message is **embedded** and the top-k relevant memories are retrieved by cosine distance and injected into the prompt. |

**PII guard:** phone numbers, emails, and card patterns are stripped before anything is persisted.

**Embeddings:** deterministic hashing embedder by default (offline, reproducible). Set `WA_EMBEDDER=openai` for real embeddings.

---

## Proof it works

`eval/golden_conversations.jsonl` holds **12 multi-turn scripts** across 7 categories. Each conversation runs **twice** — once with memory ON, once OFF:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   OVERALL (memory ON)  :  12/12  (100.0%)               │
│   OVERALL (memory OFF) :   8/12  ( 66.7%)               │
│   MEMORY DELTA         :  +33.3% accuracy               │
│   TOKEN COST           :  +105 tokens to remember       │
│                                                         │
│   By category             memory ON    memory OFF       │
│     memory_recall             3/3          0/3          │
│     tool_use                  2/2          2/2          │
│     handoff                   2/2          2/2          │
│     ...                                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The `memory_recall` row is the proof: identical conversations score **0/3 without memory** and **3/3 with it**. Results written to `eval/results/latest.json`.

---

## Quickstart

### No Docker, no API key, no internet required

```bash
pip install -r requirements.txt

python run.py --demo            # scripted multi-turn demo
python run.py                   # interactive REPL — chat yourself
python eval/run_eval.py         # scorecard: memory ON vs OFF
python -m pytest tests -q       # 29 tests, all offline
```

### Run the HTTP service

```bash
uvicorn waagent.api:app --port 8000

# Endpoints:
#   POST /chat                — simulator, synchronous
#   POST /chat/async          — enqueued, poll GET /jobs/{id}
#   POST /webhook/whatsapp    — Meta-shaped, async
#   GET  /healthz  /metrics   — health + observability
```

### Full stack (Postgres + pgvector + Redis + worker)

```bash
docker compose up --build
```

> The app auto-detects Postgres/Redis and **falls back to SQLite / in-process dict** when they're unavailable — the same code runs anywhere.

---

## Use a real model

```bash
export WA_LLM_PROVIDER=groq          # or openai
export WA_GROQ_API_KEY=...

export WA_EMBEDDER=openai            # optional: real embeddings
export WA_OPENAI_API_KEY=...
```

Prompts live in `waagent/prompts/` (planner, responder, compactor, memory classifier). Nothing else changes.

---

## Connect real WhatsApp

Set `WA_APP_SECRET`, point the Meta webhook at `/webhook/whatsapp`, and complete the `hub.challenge` verification. The payload shape, HMAC `X-Hub-Signature-256` check, and async enqueue already match the Cloud API — **no code changes needed**.

---

## Tools the agent can use

| Tool | What it does |
|------|-------------|
| `search_faq` | Keyword search over a seeded FAQ (stop-word filtered, overlap-gated) |
| `get_order_status` | Looks up order by ID from a stub DB |
| `check_refund_policy` | Returns refund rules |
| `get_session_info` | Current chat session details |
| `escalate_to_human` | Transfers to a human agent |

All tools are **read-only** and **typed** (Pydantic input models). The same registry is exposed as an [MCP server](https://modelcontextprotocol.io/) — point Claude Desktop or Cursor at `mcp_server/server.py` and drive the identical tools.

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11 |
| Agent graph | LangGraph (checkpointer + interrupts) |
| LLM | Mock (default) · Groq · OpenAI |
| Embeddings | Deterministic hashing (default) · OpenAI |
| API | FastAPI + uvicorn |
| Database | Postgres 16 + pgvector · SQLite fallback |
| Cache / Queue | Redis 7 · in-process dict fallback |
| MCP | mcp SDK, stdio transport |
| Tests | pytest (29 tests, fully offline) |

---

## Project structure

```
whatsapp-agent-fix/
├── waagent/                  # Core agent package
│   ├── agent.py              #   LangGraph graph: planner→retrieve→tools→responder→handoff
│   ├── memory.py             #   Redis sessions, compaction, pgvector semantic memory
│   ├── llm.py                #   MockLLM + OpenAI/Groq clients
│   ├── embeddings.py         #   Deterministic mock + real embedder
│   ├── tools.py              #   Typed read-only tool registry
│   ├── api.py                #   FastAPI webhook, /chat, /metrics, /healthz
│   ├── queue.py, worker.py   #   Async job processing
│   ├── ratelimit.py          #   Redis token-bucket rate limiter
│   └── prompts/              #   planner.md, responder.md, compactor.md, memory_classifier.md
├── db/
│   ├── database.py           #   SQLAlchemy engine + session helpers
│   └── migrations/           #   0001_init.sql (Postgres/pgvector schema)
├── mcp_server/
│   └── server.py             #   FastMCP server exposing the 5 tools
├── eval/
│   ├── golden_conversations.jsonl
│   ├── run_eval.py           #   Memory ON vs OFF scorecard
│   └── results/
├── tests/                    # 29 tests: memory, agent, api, queue, ratelimit, mcp
├── run.py                    # Local chat simulator (REPL + --demo)
├── docker-compose.yml        # pgvector + redis + api + worker
├── Dockerfile
└── requirements.txt
```

---

## What's real vs stubbed

| | |
|---|---|
| ✅ **Real** | LangGraph agent + interrupt handoff · Redis/pgvector memory pipeline + write policy · FastAPI webhook with HMAC signature verification + rate limiting · async queue/worker · eval harness with scorecard |
| 🔶 **Stubbed** | Order/FAQ/refund data (realistic shapes, no external system) · human-handoff destination |
| ❌ **Not included** | A real Meta number (webhook is simulator-first but Meta-shaped) · multi-tenant auth · cloud deploy docs |

> **Honesty note:** The Postgres/pgvector + Redis path is written and `docker compose config` validates, but has never been run end-to-end on this machine (Docker engine issues). Everything is verified on the SQLite/dict fallback. See `AGENTS.md` for details.

---

## Roadmap

- [ ] Shared Postgres/Redis checkpointer (horizontal scalability)
- [ ] Run Postgres/pgvector + Redis path end-to-end; add ANN index
- [ ] Real LLM mode: timeouts, retries, guardrails
- [ ] Proper queue (arq/Celery) with retries + dead-letter queue
- [ ] CI + ruff/mypy + Alembic migrations
- [ ] Webhook idempotency
- [ ] Deploy sketch: ECS + RDS + ElastiCache

---

## Documentation

| File | What it covers |
|------|---------------|
| [`WHAT_IT_DOES.md`](WHAT_IT_DOES.md) | Plain-English explanation of the project |
| [`HOW_TO_RUN.md`](HOW_TO_RUN.md) | Step-by-step run instructions (no coding) |
| [`BUILD_PLAN.md`](BUILD_PLAN.md) | Full architecture plan and milestones |
| [`AGENTS.md`](AGENTS.md) | Project status, decisions, and next steps for AI assistants |
| [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) | Production readiness assessment |

---

<div align="center">

**Built as a portfolio project for Founding Engineer — AI / Agentic Systems roles.**

*Every line of the JD maps to code in this repo.*

</div>
