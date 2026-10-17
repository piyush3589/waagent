from waagent.config import settings
from waagent.memory import (
    SessionStore,
    SemanticMemory,
    classify_memory,
    compact,
    needs_compaction,
    strip_pii,
)
from waagent.schemas import MemoryKind


def test_write_policy_remembers_durable_facts():
    assert classify_memory("I prefer Hindi for support") == MemoryKind.preference
    assert classify_memory("My order 1234 has failed twice now") == MemoryKind.event
    assert classify_memory("I am very frustrated with this issue") == MemoryKind.sentiment
    assert classify_memory("My name is Asha and I live in Pune") == MemoryKind.fact


def test_write_policy_rejects_transient():
    assert classify_memory("hi") is None
    assert classify_memory("thanks!") is None
    assert classify_memory("What is my refund status?") is None


def test_pii_is_stripped():
    out = strip_pii("call me on 9876543210 or mail a@b.com")
    assert "9876543210" not in out
    assert "a@b.com" not in out
    assert "[phone]" in out and "[email]" in out


def test_session_turn_buffer_and_compaction():
    session = SessionStore()
    for i in range(settings.summary_trigger_turns + 3):
        session.append_turn("u1", "user", f"message {i}")
        session.append_turn("u1", "assistant", f"reply {i}")
    assert needs_compaction(session, "u1")
    summary = compact(session, "u1")
    assert summary
    assert session.get_summary("u1") == summary
    assert len(session.get_turns("u1")) <= max(2, settings.session_max_turns // 2)


def test_semantic_remember_and_recall():
    mem = SemanticMemory()
    assert mem.remember("u2", "I prefer Hindi and WhatsApp for updates") is not None
    assert mem.remember("u2", "My name is Ravi and I live in Mumbai") is not None
    # questions are not stored as memory
    assert mem.remember("u2", "What is my order status?") is None

    hits = mem.recall("u2", "which language do I prefer for updates", k=3)
    assert hits
    assert any("Hindi" in h.content for h in hits)
    # isolation between users
    assert mem.recall("someone-else", "language preference") == []
