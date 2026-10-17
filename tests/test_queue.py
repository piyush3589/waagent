from waagent.queue import JobQueue


def test_enqueue_is_queued_then_drained():
    q = JobQueue()  # offline: no redis, no thread
    job_id = q.enqueue({"user_id": "q1", "text": "where is my order 1234"})
    assert q.get(job_id)["status"] == "queued"

    processed = q.drain_once(lambda payload: {"reply": "ok", "user_id": payload["user_id"]})
    assert processed is True

    job = q.get(job_id)
    assert job["status"] == "done"
    assert job["result"]["reply"] == "ok"


def test_drain_once_empty_returns_false():
    q = JobQueue()
    assert q.drain_once(lambda payload: {}, timeout=0.05) is False


def test_handler_error_is_captured():
    q = JobQueue()

    def boom(_payload):
        raise RuntimeError("kaboom")

    job_id = q.enqueue({"user_id": "q2", "text": "hi"})
    q.drain_once(boom)
    job = q.get(job_id)
    assert job["status"] == "error"
    assert "kaboom" in job["result"]["error"]
