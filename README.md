# WA-Agent

**A WhatsApp-style customer-support agent with short-term and long-term memory, tool use, human handoff, and an evaluation harness.**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-black.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

WA-Agent handles each incoming message with a LangGraph state machine that **plans, retrieves memory, calls tools, and responds**, and it hands the conversation to a human when confidence is low or the user asks for one. It remembers user preferences across conversations, strips basic PII before anything is stored, and ships with a harness that scores the agent with memory ON versus OFF.

It runs **fully offline by default** (mock LLM, deterministic embeddings, SQLite and in-process fallbacks), so you can clone it, run it, and test it without an API key.

---

## Contents

- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Memory pipeline](#memory-pipeline)
- [Evaluation](#evaluation)
- [Quickstart](#quickstart)
- [Running with a real model](#running-with-a-real-model)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Tools and MCP server](#tools-and-mcp-server)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Status: what is real, stubbed, and untested](#status-what-is-real-stubbed-and-untested)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Demo

Output of `python run.py --demo` (mock LLM, no API key):

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

## Features

- **Agent graph.** A LangGraph state machine with typed state, a checkpointer, and an `interrupt()`-based human handoff.
- **Two-tier memory.** A short-term session buffer with rolling-summary compaction, and long-term semantic memory with vector retrieval.
- **Write policy.** A pure, unit-tested function decides which facts are worth persisting (durable, user-specific, non-PII).
- **Typed tools.** A read-only registry of Pydantic-validated tools, also exposed over the Model Context Protocol (MCP).
- **Async webhook.** A FastAPI endpoint with HMAC signature verification, per-user rate limiting, and queue-backed processing off the request path.
- **Offline-first.** Mock LLM, deterministic embeddings, and automatic SQLite and in-process fallbacks.
- **Evaluation harness.** Golden conversations scored with memory ON versus OFF, reporting accuracy and token cost.

---

## Architecture

```
               WhatsApp Cloud API  /  local simulator
                          |
               +----------v-----------+     signature check,
               |   WEBHOOK (FastAPI)  |     rate limit,
               |  POST /webhook/...   |     enqueue -> 202
               +----------+-----------+
                          |  queue
               +----------v-----------+
               |       WORKER         |     agent runs off
               +----------+-----------+     the request path
                          v
               +----------------------+
               |   LANGGRAPH AGENT    |
               |                      |
               |   planner            |  classify intent, select tools
               |   retrieve           |  load summary and top-k memories
               |   tools              |  run read-only lookups
               |   responder          |  compose reply and confidence
               |   human_handoff      |  interrupt() for a human
               +----------+-----------+
       +------------------+------------------+
       v                  v                   v
 Session store      Vector memories       Audit tables
 (turns+summary)    (semantic recall)     (messages, traces)
```

Each message follows the same loop: plan, retrieve memory, call tools, respond. If reply confidence is below `WA_MIN_REPLY_CONFIDENCE`, or the user asks for a person, the graph interrupts and pauses.

At startup the app looks for Postgres and Redis and falls back to SQLite and an in-process dictionary when they are unavailable, so the same code runs in any environment.

---

## Memory pipeline

| Tier | Backend | What it does |
| --- | --- | --- |
| Short-term | Redis (dict fallback) | Keeps the last *N* turns per user. When the buffer grows past a threshold, older turns are folded into a rolling summary so context stays bounded. |
| Long-term | Postgres + pgvector (SQLite fallback) | A write policy extracts durable, user-specific facts (`preference`, `fact`, `event`, `sentiment`). Each new message is embedded and the top-*k* memories are retrieved by cosine distance and injected into the responder prompt. |

- **PII guard.** Phone numbers, email addresses, and card-number patterns are stripped before anything is persisted. This is pattern-based; see [Known limitations](#known-limitations).
- **Embeddings.** A deterministic hashing embedder by default, so evaluations are reproducible. A real embedder is opt-in via `WA_EMBEDDER`.
- **Decay.** Memories track an access count and can be archived when unused.

---

## Evaluation

`eval/golden_conversations.jsonl` holds 12 multi-turn conversations across 7 categories. `eval/run_eval.py` runs each one twice, with memory ON and OFF, and reports accuracy and token cost.

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

**How to read these numbers.** They come from the deterministic mock LLM and hashing embedder. That makes them reproducible and fast, but it means they measure whether the *memory plumbing* works (facts are written, retrieved, and injected), not how a real model behaves. The suite is small (12 conversations), runs once with no variance estimate, and the `memory_recall` and `pii` rows are close to correct by construction under a mock.

### Mock LLM vs. a real model

The same harness was also run against `openai/gpt-oss-120b` on Groq (17 Oct 2026):

| | Memory ON | Memory OFF | Delta |
| --- | --- | --- | --- |
| Mock LLM | 12/12 (100.0%) | 8/12 (66.7%) | +33.3% |
| Groq (real model) | 8/12 (66.7%) | 6/12 (50.0%) | +16.7% |

Groq by category (memory ON / OFF): billing 1/1, 1/1; faq 2/2, 2/2; handoff 1/2, 1/2; memory_recall 1/3, 0/3; pii 1/1, 0/1; smalltalk 0/1, 0/1; tool_use 2/2, 2/2.

What this shows and does not show:

- Memory still helps on a real model, but by less than the mock suggests (+16.7% versus +33.3%).
- Groq runs are **not reproducible**: answers vary between runs, so these numbers are a single sample.
- The harness checks exact substrings (`reply_contains`). Three of the four Groq failures are replies that omit an expected word ("Hindi", "help"), which penalises correct free-form answers. This makes the harness noisy for real models.
- One failure is a real planner miss: the off-topic conversation did not trigger a handoff.
- Token counts are estimated locally, not taken from Groq usage data.
- The first Groq run failed with HTTP 429; retry with exponential backoff for 429 and 5xx responses was then added to `OpenAICompatibleLLM._chat` in `waagent/llm.py`.

Full machine-readable results are written to `eval/results/latest.json`.

---

## Quickstart

Requires Python 3.11. No Docker, API key, or network needed.

```bash
pip install -r requirements.txt

python run.py --demo            # scripted multi-turn demo (memory + handoff)
python run.py                   # interactive REPL
python eval/run_eval.py         # memory ON vs OFF scorecard
python -m pytest tests -q       # 29 tests, offline
```

Run the HTTP service:

```bash
uvicorn waagent.api:app --port 8000
```

Full stack (Postgres + pgvector + Redis + worker):

```bash
docker compose up --build
```

The schema in `db/migrations/0001_init.sql` is applied automatically, and `api` and `worker` share one image. **This path has not been run end-to-end yet** (see [Status](#status-what-is-real-stubbed-and-untested)).

---

## Running with a real model

The agent talks to any OpenAI-compatible `/chat/completions` endpoint. Copy `.env.example` to `.env` and set:

```bash
WA_LLM_PROVIDER=groq                      # or: openai
WA_GROQ_API_KEY=your_key_here
WA_LLM_MODEL=openai/gpt-oss-120b          # any model on your account

# Optional: real embeddings instead of the deterministic mock
WA_EMBEDDER=openai
WA_OPENAI_API_KEY=your_key_here
```

No code changes are needed. Prompts in `waagent/prompts/` drive planning, response, compaction, and memory classification.

**Status of real-model mode.** Evaluated on Groq with `openai/gpt-oss-120b` (see [Evaluation](#evaluation) for the results). Retries with exponential backoff are applied to 429 and 5xx responses. Two known limits apply in this mode:

- The responder returns a fixed confidence of 0.8, so the low-confidence handoff does not trigger.
- If a model response fails to parse, the planner silently falls back to the mock planner.

---

## Configuration

All settings use the `WA_` prefix and are read from the environment or a local `.env` file (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `WA_LLM_PROVIDER` | `mock` | `mock`, `groq`, or `openai` |
| `WA_LLM_MODEL` | `openai/gpt-oss-120b` | Model name for the chosen provider |
| `WA_EMBEDDER` | `mock` | `mock` or `openai` |
| `WA_DATABASE_URL` | local Postgres | SQLAlchemy database URL |
| `WA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `WA_SESSION_MAX_TURNS` | `8` | Turn buffer size before compaction |
| `WA_SUMMARY_TRIGGER_TURNS` | `6` | Turns before the summary is rewritten |
| `WA_MEMORY_TOP_K` | `5` | Memories retrieved per turn |
| `WA_MIN_REPLY_CONFIDENCE` | `0.55` | Below this, the agent hands off |
| `WA_RATE_LIMIT_PER_MINUTE` | `20` | Per-user request limit (token bucket) |
| `WA_APP_SECRET` | empty | Meta app secret for webhook HMAC checks |

---

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/chat` | Synchronous simulator endpoint |
| `POST` | `/chat/async` | Enqueued; poll `GET /jobs/{id}` for the result |
| `POST` | `/webhook/whatsapp` | Meta-shaped webhook (async) |
| `GET` | `/webhook/whatsapp` | Webhook verification (`hub.challenge`) |
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Request counts, latency, handoff and memory-hit rates |

To connect the WhatsApp Cloud API, set `WA_APP_SECRET`, point the Meta webhook at `/webhook/whatsapp`, and complete the verification handshake. The payload shape, `X-Hub-Signature-256` check, and async enqueue are written to match the Cloud API, but the integration has not been tested against a live Meta number.

---

## Tools and MCP server

All tools are typed, read-only, and validated with Pydantic input models.

| Tool | Description |
| --- | --- |
| `search_faq` | Keyword search over a seeded FAQ (stop-word filtered, overlap-gated) |
| `get_order_status` | Looks up an order by ID |
| `check_refund_policy` | Returns the refund policy |
| `get_session_info` | Returns current session details |
| `escalate_to_human` | Transfers the conversation to a human agent |

The same registry is exposed as a Model Context Protocol server (`mcp_server/server.py`, stdio transport), so any MCP client such as Claude Desktop or Cursor can call the identical tools.

---

## Project structure

```
waagent/
├── waagent/                  # Core package
│   ├── agent.py              #   LangGraph graph: planner -> retrieve -> tools -> responder -> handoff
│   ├── memory.py             #   Sessions, summary compaction, semantic memory
│   ├── llm.py                #   MockLLM + OpenAI-compatible client (Groq / OpenAI)
│   ├── embeddings.py         #   Deterministic mock + real embedder
│   ├── tools.py              #   Typed read-only tool registry
│   ├── api.py                #   FastAPI webhook, /chat, /metrics, /healthz
│   ├── queue.py, worker.py   #   Async job processing
│   ├── ratelimit.py          #   Token-bucket rate limiter
│   ├── schemas.py            #   Pydantic contracts
│   └── prompts/              #   planner, responder, compactor, memory_classifier
├── db/
│   ├── database.py           #   SQLAlchemy engine and session helpers
│   └── migrations/           #   0001_init.sql (Postgres/pgvector schema)
├── mcp_server/server.py      # FastMCP server exposing the 5 tools
├── eval/                     # Golden conversations, run_eval.py, results/
├── tests/                    # 29 tests: memory, agent, api, queue, ratelimit, mcp
├── run.py                    # Local chat simulator (REPL + --demo)
├── docker-compose.yml        # pgvector + redis + api + worker
├── Dockerfile
└── requirements.txt
```

---

## Testing

```bash
python -m pytest tests -q      # 29 tests
```

The suite covers memory (compaction, write policy, PII stripping, retrieval), the end-to-end agent loop, the API, the queue and worker, the rate limiter, and the MCP server. All tests run offline against the mock LLM.

---

## Status: what is real, stubbed, and untested

| | |
| --- | --- |
| **Working and verified (SQLite / in-process fallback)** | LangGraph agent and interrupt handoff; memory pipeline and write policy; FastAPI endpoints; webhook HMAC verification and rate limiting; async queue and worker; evaluation harness; MCP server |
| **Written but not run end-to-end** | The Postgres + pgvector + Redis path. `docker compose config` validates, but the full stack has not been started (Docker was unavailable), so all verified results come from the fallbacks |
| **Stubbed** | Order, FAQ, and refund data in `waagent/tools.py` (realistic shapes, no external system); the human-handoff destination |
| **Not included** | A live Meta WhatsApp number, multi-tenant authentication, cloud deployment tooling |

---

## Known limitations

- **Evaluation is small and noisy.** The suite has 12 conversations. The mock-LLM scores are reproducible but only exercise the memory plumbing; the Groq scores are a single non-reproducible sample, and the exact-substring checks penalise correct free-form replies.
- **Handoff on off-topic input is unreliable with a real model.** In the Groq run, the off-topic conversation did not trigger a handoff.
- **Fixed confidence in real-model mode.** The Groq responder returns 0.8, so confidence-based handoff does not trigger there.
- **Silent fallback.** A failed parse in real-model mode falls back to the mock planner without a warning.
- **PII guard is pattern-based.** It catches phone numbers, emails, and card-number patterns. It does not detect names, addresses, or identifiers written in free text, so it is not sufficient on its own for regulated data such as KYC details.
- **Sales flows are not implemented.** The agent is a support agent (FAQs, order lookup, refunds, handoff). Intent qualification, follow-ups, and cross-sell are not built.

---

## Roadmap

- [ ] Run the Postgres + pgvector + Redis path end-to-end; add an ANN index
- [ ] Repeat real-LLM evals with a variance estimate; grow the golden set; replace exact-substring checks with looser or LLM-graded checks
- [ ] Real confidence scoring for the Groq responder; surface the mock-planner fallback instead of hiding it
- [ ] Name and address detection in the PII guard (NER or LLM-based)
- [ ] Shared Postgres/Redis checkpointer and distributed rate limiter
- [ ] A proper queue (arq or Celery) with retries and a dead-letter queue
- [ ] Webhook idempotency
- [ ] CI (pytest, ruff, mypy) and Alembic migrations
- [ ] Deployment sketch (ECS + RDS + ElastiCache)

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

