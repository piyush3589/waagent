import time

from fastapi.testclient import TestClient

from waagent.api import app, get_limiter

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_order_status():
    resp = client.post("/chat", json={"user_id": "api-u1", "text": "where is my order 1234"})
    assert resp.status_code == 200
    body = resp.json()
    assert "get_order_status" in body["tools_used"]
    assert "packed" in body["reply"].lower()


def test_webhook_verification_handshake():
    resp = client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "wa-verify-token", "hub.challenge": "chal-123"},
    )
    assert resp.status_code == 200
    assert resp.text == "chal-123"


def test_webhook_inbound_message():
    payload = {
        "entry": [{"changes": [{"value": {"messages": [{"from": "w1", "text": {"body": "hello"}}]}}]}]
    }
    resp = client.post("/webhook/whatsapp", json=payload)
    assert resp.status_code == 202
    jobs = resp.json()["jobs"]
    assert jobs and jobs[0]["status"] == "accepted"
    assert jobs[0]["job_id"]


def test_async_chat_job_result():
    resp = client.post("/chat/async", json={"user_id": "async-u1", "text": "where is my order 1234"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = {}
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") == "done":
            break
        time.sleep(0.1)
    assert job["status"] == "done"
    assert "get_order_status" in job["result"]["tools_used"]


def test_async_unknown_job_404():
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "counters" in resp.json()


def test_rate_limit_returns_429():
    limiter = get_limiter()
    original = limiter.per_minute
    limiter.per_minute = 1
    try:
        client.post("/chat", json={"user_id": "rl-user", "text": "hi"})
        resp = client.post("/chat", json={"user_id": "rl-user", "text": "hi"})
        assert resp.status_code == 429
    finally:
        limiter.per_minute = original
