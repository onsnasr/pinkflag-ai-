"""Voice call endpoint with fake ears/voice (no models, no mic, no internet)."""
import base64

from fastapi.testclient import TestClient

from pinkflag.agent.memory import Memory
from pinkflag.app.server import create_app
from pinkflag.voice import speakable


def gemini(text):
    def factory():
        return lambda body: {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
    return factory


def test_voice_round_trip():
    heard = []
    app = create_app(Memory(":memory:"), gemini("okay babe **breathe** 💕"), detector=(None, None),
                     stt=lambda audio: {"text": "he yelled at me", "language": "en"},
                     tts=lambda text: heard.append(text) or b"RIFFfake")
    client = TestClient(app)
    gid = client.post("/api/guys", json={"name": "Amine"}).json()["id"]
    res = client.post(f"/api/guys/{gid}/voice", files={"audio": ("v.webm", b"\x00\x01", "audio/webm")})
    body = res.json()
    assert res.status_code == 200
    assert body["transcript"] == "he yelled at me" and "breathe" in body["reply"]
    assert base64.b64decode(body["audio"]) == b"RIFFfake"
    assert heard == ["okay babe **breathe** 💕"]
    assert [m["role"] for m in client.get(f"/api/guys/{gid}/chat").json()] == ["user", "model"]


def test_silence_gets_a_gentle_retry():
    app = create_app(Memory(":memory:"), gemini("unused"), detector=(None, None),
                     stt=lambda audio: {"text": "", "language": None}, tts=lambda text: b"x")
    client = TestClient(app)
    gid = client.post("/api/guys", json={"name": "K"}).json()["id"]
    body = client.post(f"/api/guys/{gid}/voice", files={"audio": ("v.webm", b"", "audio/webm")}).json()
    assert "didn't catch" in body["reply"]


def test_greeting_is_synthesized_once():
    calls = []
    app = create_app(Memory(":memory:"), gemini("x"), detector=(None, None),
                     stt=lambda a: {"text": ""}, tts=lambda t: calls.append(t) or b"wav")
    client = TestClient(app)
    first, second = client.get("/api/voice/greeting").json(), client.get("/api/voice/greeting").json()
    assert first["text"].startswith("Hello bestie") and first == second
    assert len(calls) == 1


def test_speakable_strips_emojis_and_markdown():
    assert speakable("heyy **bestie** 💕 that's 🚩") == "heyy bestie that's"


def test_speakable_reads_arabizi_numbers_as_letters():
    assert speakable("ya o5ti, 3lech?") == "ya okhti, alech?"
    assert speakable("it was 2 am, he sent 5 texts") == "it was 2 am, he sent 5 texts"
