"""The web API, tested with a fake Gemini and a tiny real detector (no keys, no internet)."""
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from pinkflag.agent.memory import Memory
from pinkflag.agent.vibe import summary, vibe_score
from pinkflag.app.server import create_app
from pinkflag.data import load_messages
from pinkflag.models.baseline import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def detector():
    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    df = load_messages(ROOT / cfg["data"]["path"])
    return build_pipeline(cfg).fit(df.text, df.label), "test-model"


def scripted_gemini(*responses):
    """Each call to the factory gets a transport that replays the responses in order."""
    script = list(responses)

    def factory():
        def send(body):
            return script.pop(0)
        return send
    return factory


def text(t):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t}]}}]}


def call(name, args):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": name, "args": args}}]}}]}


def test_vibe_score():
    assert vibe_score([]) is None
    assert vibe_score([{"label": "green_flag"}] * 3) == 0
    assert vibe_score([{"label": "intimidation"}]) == 100
    s = summary([{"label": "green_flag"}, {"label": "controlling"}])
    assert s["red_flags"] == 1 and s["green_flags"] == 1 and 0 < s["score"] < 100


def test_full_chat_flow(detector):
    gemini = scripted_gemini(
        call("analyze_messages", {"messages": ["send me ur location rn and who are you with"]}),
        call("update_profile", {"field": "works_or_studies", "value": "med student"}),
        call("remember", {"note": "talking for 2 months"}),
        text("bestie that's **controlling** ⛓️"),
    )
    client = TestClient(create_app(Memory(":memory:"), gemini, detector=detector))

    guy = client.post("/api/guys", json={"name": "Amine"}).json()
    assert guy["score"] is None and guy["verdict"] == "no receipts yet"

    res = client.post(f"/api/guys/{guy['id']}/chat", json={"text": "he said: send me ur location"})
    body = res.json()
    assert res.status_code == 200
    assert "controlling" in body["reply"]
    assert len(body["receipts"]) == 1 and body["receipts"][0]["text"].startswith("send me")
    assert body["guy"]["works_or_studies"] == "med student"
    assert body["guy"]["analysed"] == 1

    receipts = client.get(f"/api/guys/{guy['id']}/receipts").json()
    assert receipts["notes"] == ["talking for 2 months"]
    assert len(receipts["timeline"]) == 1

    history = client.get(f"/api/guys/{guy['id']}/chat").json()
    assert [m["role"] for m in history] == ["user", "model"]


def test_roster_best_picks_lowest_score():
    mem = Memory(":memory:")
    client = TestClient(create_app(mem, scripted_gemini(), detector=(None, None)))
    a = client.post("/api/guys", json={"name": "Amine"}).json()["id"]
    k = client.post("/api/guys", json={"name": "Karim"}).json()["id"]
    assert client.get("/api/roster/best").status_code == 404  # no receipts yet
    mem.add_analyses(a, [{"message": "x", "label": "controlling", "confidence": 0.9}])
    mem.add_analyses(k, [{"message": "y", "label": "green_flag", "confidence": 0.8}])
    assert client.get("/api/roster/best").json()["name"] == "Karim"


def test_update_and_delete_guy():
    client = TestClient(create_app(Memory(":memory:"), scripted_gemini(), detector=(None, None)))
    g = client.post("/api/guys", json={"name": "Youssef"}).json()
    upd = client.patch(f"/api/guys/{g['id']}", json={"status": "situationship", "age": "23"}).json()
    assert upd["status"] == "situationship" and upd["age"] == "23"
    assert client.delete(f"/api/guys/{g['id']}").status_code == 204
    assert client.get("/api/guys").json() == []
    assert client.get(f"/api/guys/{g['id']}/chat").status_code == 404

def test_two_year_followup_revises_pending_receipt(detector):
    quote = "I love you very very much"
    gemini = scripted_gemini(
        call("analyze_messages", {"messages": [quote]}),
        text("How long have you been dating?"),
        call("revise_rating", {"receipt_id": 1, "label": "green_flag"}),
        text("Given that context, the message sounds affectionate."),
    )
    client = TestClient(create_app(Memory(":memory:"), gemini, detector=detector))
    gid = client.post("/api/guys", json={"name": "Nour"}).json()["id"]
    path = f"/api/guys/{gid}"
    first = client.post(path + "/chat", json={"text": f"He said: {quote}"}).json()
    assert first["receipts"][0]["label"] == "needs_context"
    assert first["guy"]["score"] is None
    assert first["guy"]["red_flags"] == first["guy"]["green_flags"] == 0
    final = client.post(path + "/chat", json={"text": "Two years. It felt affectionate."}).json()
    assert final["guy"]["analysed"] == final["guy"]["green_flags"] == 1
    assert final["guy"]["score"] == final["guy"]["needs_context"] == 0
    assert client.patch(path + "/receipts/1", json={"label": "needs_context"}).is_success
    assert client.get(path + "/receipts").json()["guy"]["score"] is None
    assert client.patch("/api/guys/999/receipts/1", json={"label": "green_flag"}).status_code == 404
