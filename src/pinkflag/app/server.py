"""The PinkFlag API.

    GET    /                              the chat app (web/app/index.html)
    GET    /api/guys                      the roster: every guy + his vibe score
    POST   /api/guys                      add a guy            {name, status?, emoji?}
    PATCH  /api/guys/{id}                 edit status/profile  {status?, works_or_studies?, ...}
    DELETE /api/guys/{id}                 delete a guy and everything about him
    GET    /api/guys/{id}/chat            the conversation with Sisi about him
    POST   /api/guys/{id}/chat            talk to Sisi          {text} -> {reply, receipts, guy}
    GET    /api/guys/{id}/receipts        vibe meter, stats, timeline, what Sisi remembers
    GET    /api/roster/best               who the wheel lands on (lowest vibe score)

Everything stays on this computer: one SQLite file (data/agent/sisi.db).
"""
import base64
from collections.abc import Callable
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinkflag.agent.memory import Memory
from pinkflag.agent.sisi import Sisi, Transport
from pinkflag.agent.tools import Toolbox, load_detector
from pinkflag.agent.vibe import summary
from pinkflag.taxonomy import LABELS

WEB_DIR = Path(__file__).resolve().parents[3] / "web" / "app"


class NewGuy(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    status: str = "talking stage"
    emoji: str = "💌"


class GuyUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    emoji: str | None = None
    works_or_studies: str | None = None
    met_at: str | None = None
    age: str | None = None
    detail: str | None = None


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def _receipt(r: dict) -> dict:
    lab = LABELS.get(r["label"])
    return {"text": r.get("message") or r.get("text"), "label": r["label"],
            "name": lab.display if lab else r["label"], "emoji": lab.emoji if lab else "",
            "is_red_flag": lab.is_red_flag if lab else True,
            "confidence": r["confidence"], "created_at": r.get("created_at")}


def create_app(memory: Memory, transport_factory: Callable[[], Transport],
               detector=None, history_turns: int = 30,
               stt: Callable[[bytes], dict] | None = None,
               tts: Callable[[str], bytes] | None = None) -> FastAPI:
    app = FastAPI(title="PinkFlag", docs_url="/api/docs")
    detector = detector or load_detector()

    def _stt(audio: bytes) -> dict:
        if stt:
            return stt(audio)
        from pinkflag.voice import transcribe  # heavy: only loaded when someone calls Sisi
        return transcribe(audio)

    def _tts(text: str) -> bytes:
        if tts:
            return tts(text)
        from pinkflag.voice import synthesize
        return synthesize(text)

    def get_guy(guy_id: int) -> dict:
        guy = memory.guy(guy_id)
        if guy is None:
            raise HTTPException(404, "guy not found")
        return guy

    def with_summary(guy: dict) -> dict:
        return {**guy, **summary(memory.analyses(guy["id"]))}

    @app.get("/api/guys")
    def list_guys():
        return [with_summary(g) for g in memory.guys()]

    @app.post("/api/guys", status_code=201)
    def add_guy(body: NewGuy):
        return with_summary(get_guy(memory.add_guy(body.name, body.status, body.emoji)))

    @app.patch("/api/guys/{guy_id}")
    def update_guy(guy_id: int, body: GuyUpdate):
        get_guy(guy_id)
        memory.update_guy(guy_id, **body.model_dump(exclude_none=True))
        return with_summary(get_guy(guy_id))

    @app.delete("/api/guys/{guy_id}", status_code=204)
    def delete_guy(guy_id: int):
        get_guy(guy_id)
        memory.delete_guy(guy_id)

    @app.get("/api/guys/{guy_id}/chat")
    def chat_history(guy_id: int):
        get_guy(guy_id)
        return [{"role": role, "text": text}
                for role, text in memory.history(f"guy-{guy_id}", limit=200)]

    @app.post("/api/guys/{guy_id}/chat")
    def chat(guy_id: int, body: ChatIn):
        get_guy(guy_id)
        tools = Toolbox(memory, detector=detector, guy_id=guy_id)
        sisi = Sisi(transport_factory(), memory, tools, session=f"guy-{guy_id}",
                    history_turns=history_turns)
        try:
            reply = sisi.reply(body.text)
        except (requests.RequestException, KeyError, ValueError):
            return JSONResponse(status_code=503, content={
                "reply": "ugh my brain's servers are busy rn 😵‍💫 try again in a minute, bestie?",
                "receipts": [], "guy": with_summary(get_guy(guy_id))})
        return {"reply": reply, "receipts": [_receipt(r) for r in tools.analysed],
                "guy": with_summary(get_guy(guy_id))}

    @app.post("/api/transcribe")
    async def transcribe(audio: UploadFile):
        """🎙️ Dictation in the chat box: your voice -> text (Whisper, local)."""
        return _stt(await audio.read())

    greeting_cache: dict[str, str] = {}

    @app.get("/api/voice/greeting")
    def greeting():
        """What Sisi says when she picks up (synthesized once, then cached)."""
        text = "Hello bestie! I'm here, talk to me. What did he do?"
        if text not in greeting_cache:
            greeting_cache[text] = base64.b64encode(_tts(text)).decode()
        return {"text": text, "audio": greeting_cache[text]}

    @app.post("/api/guys/{guy_id}/voice")
    async def voice(guy_id: int, audio: UploadFile):
        """📞 Call Sisi: your voice in -> Whisper -> Sisi -> Piper -> her voice out."""
        get_guy(guy_id)
        heard = _stt(await audio.read())
        if not heard["text"]:
            reply, receipts = "sorry babe, I didn't catch that. say it again?", []
        else:
            tools = Toolbox(memory, detector=detector, guy_id=guy_id)
            sisi = Sisi(transport_factory(), memory, tools, session=f"guy-{guy_id}",
                        history_turns=history_turns)
            try:
                reply = sisi.reply(heard["text"], voice=True)
            except (requests.RequestException, KeyError, ValueError):
                reply = "ugh, my brain's servers are busy. try again in a minute?"
            receipts = [_receipt(r) for r in tools.analysed]
        return {"transcript": heard["text"], "language": heard.get("language"), "reply": reply,
                "receipts": receipts, "audio": base64.b64encode(_tts(reply)).decode()}

    @app.get("/api/guys/{guy_id}/receipts")
    def receipts(guy_id: int):
        guy = with_summary(get_guy(guy_id))
        return {"guy": guy, "timeline": [_receipt(a) for a in memory.analyses(guy_id)],
                "notes": memory.notes(f"guy-{guy_id}")}

    @app.get("/api/roster/best")
    def best():
        scored = [g for g in (with_summary(g) for g in memory.guys()) if g["score"] is not None]
        if not scored:
            raise HTTPException(404, "no receipts yet: tell Sisi what they said first")
        return min(scored, key=lambda g: g["score"])

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app
