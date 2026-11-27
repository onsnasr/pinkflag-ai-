"""Public hosting safety: visitors are isolated, have a daily limit, and can delete their data."""
from fastapi.testclient import TestClient

from pinkflag.agent.memory import Memory
from pinkflag.app.server import create_app

A = {"X-PinkFlag-User": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}
B = {"X-PinkFlag-User": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"}


def gemini():
    return lambda body: {"candidates": [{"content": {"role": "model", "parts": [{"text": "hey 💕"}]}}]}


def client(**kw):
    return TestClient(create_app(Memory(":memory:"), lambda: gemini(), detector=(None, None), **kw))


def test_visitors_never_see_each_other():
    c = client()
    gid = c.post("/api/guys", json={"name": "Amine"}, headers=A).json()["id"]
    assert [g["name"] for g in c.get("/api/guys", headers=A).json()] == ["Amine"]
    assert c.get("/api/guys", headers=B).json() == []
    for method, path in [("get", "chat"), ("get", "receipts"), ("post", "chat")]:
        kwargs = {"json": {"text": "hi"}} if method == "post" else {}
        assert getattr(c, method)(f"/api/guys/{gid}/{path}", headers=B, **kwargs).status_code == 404
    assert c.delete(f"/api/guys/{gid}", headers=B).status_code == 404


def test_daily_limit_per_visitor():
    c = client(daily_limit=2)
    gid = c.post("/api/guys", json={"name": "K"}, headers=A).json()["id"]
    assert c.post(f"/api/guys/{gid}/chat", json={"text": "1"}, headers=A).status_code == 200
    assert c.post(f"/api/guys/{gid}/chat", json={"text": "2"}, headers=A).status_code == 200
    blocked = c.post(f"/api/guys/{gid}/chat", json={"text": "3"}, headers=A)
    assert blocked.status_code == 429 and "come back tomorrow" in blocked.json()["reply"]
    gid_b = c.post("/api/guys", json={"name": "Y"}, headers=B).json()["id"]
    assert c.post(f"/api/guys/{gid_b}/chat", json={"text": "hi"}, headers=B).status_code == 200


def test_delete_my_data():
    c = client()
    c.post("/api/guys", json={"name": "Amine"}, headers=A)
    c.post("/api/guys", json={"name": "Karim"}, headers=A)
    c.post("/api/guys", json={"name": "Youssef"}, headers=B)
    assert c.delete("/api/me", headers=A).json()["deleted_guys"] == 2
    assert c.get("/api/guys", headers=A).json() == []
    assert len(c.get("/api/guys", headers=B).json()) == 1


def test_bad_visitor_id_falls_back_to_local():
    c = client()
    c.post("/api/guys", json={"name": "X"}, headers={"X-PinkFlag-User": "../../etc"})
    assert len(c.get("/api/guys").json()) == 1  # treated as 'local', no injection


def test_security_headers():
    r = client().get("/api/guys")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
