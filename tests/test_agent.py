from waagent.agent import ConversationAgent
from waagent.memory import SessionStore
from waagent.schemas import Intent
from waagent.tools import SearchFaqIn, search_faq


def _agent(**kwargs) -> ConversationAgent:
    return ConversationAgent(session=SessionStore(), **kwargs)


def test_tool_use_order_status():
    agent = _agent()
    result = agent.run("u-orders", "where is my order 1234")
    assert "get_order_status" in result.reply.tools_used
    assert "packed" in result.reply.text.lower()
    assert result.reply.confidence >= 0.9


def test_memory_used_on_later_turn():
    agent = _agent()
    agent.run("u-mem", "I prefer Hindi and WhatsApp for updates")
    result = agent.run("u-mem", "which language should you reply in for updates")
    assert result.retrieved, "expected a recalled memory"
    assert "remember" in result.reply.text.lower()


def test_handoff_on_human_request():
    agent = _agent()
    result = agent.run("u-handoff", "I want to talk to a real person please")
    assert result.reply.handoff is True
    assert result.plan.intent == Intent.handoff_request


def test_memory_can_be_disabled():
    agent = _agent(memory_enabled=False)
    assert agent.semantic is None
    result = agent.run("u-nomem", "I prefer Hindi and WhatsApp for updates")
    assert result.retrieved == []
    assert result.written == []


def test_faq_tool_selected():
    agent = _agent()
    result = agent.run("u-faq", "how long does delivery take")
    assert "search_faq" in result.reply.tools_used


def test_interrupt_handoff_path_and_recovery():
    agent = ConversationAgent(session=SessionStore(), use_interrupt=True)
    result = agent.run("u-intr", "I want to talk to a human")
    assert result.reply.handoff is True
    assert "human" in result.reply.text.lower()
    # the paused thread resumes cleanly for the next turn
    nxt = agent.run("u-intr", "where is my order 1234")
    assert nxt.reply.handoff is False
    assert "packed" in nxt.reply.text.lower()


def test_search_faq_ignores_stopword_only_overlap():
    out = search_faq(SearchFaqIn(query="what is the capital of france?"), {})
    assert out["matches"] == [], "stopword-only overlap must not match an FAQ"


def test_search_faq_still_matches_real_questions():
    out = search_faq(SearchFaqIn(query="how long does delivery take"), {})
    assert out["matches"], "a real delivery question must still match"


def test_off_topic_question_is_not_answered_as_faq():
    agent = _agent()
    result = agent.run("u-offtopic", "what is the capital of france?")
    text = result.reply.text.lower()
    assert "refund" not in text
    assert "policy" not in text
    assert "7 days" not in text
    assert result.reply.handoff is True
