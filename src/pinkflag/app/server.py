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
import os
import re
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path

import requests
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
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


class RatingUpdate(BaseModel):
    label: str


def _receipt(r: dict) -> dict:
    lab = LABELS.get(r["label"])
    return {"id": r.get("id"), "text": r.get("message") or r.get("text"), "label": r["label"],
            "name": lab.display if lab else "Needs context", "emoji": lab.emoji if lab else "",
            "is_red_flag": lab.is_red_flag if lab else None,
            "confidence": r["confidence"], "created_at": r.get("created_at")}


CURRENT_OWNER: ContextVar[str] = ContextVar("owner", default="local")
VISITOR_ID = re.compile(r"[A-Za-z0-9-]{16,64}")


def create_app(memory: Memory, transport_factory: Callable[[], Transport],
               detector=None, history_turns: int = 30,
               stt: Callable[[bytes], dict] | None = None,
               tts: Callable[[str], bytes] | None = None,
               daily_limit: int | None = None) -> FastAPI:
    app = FastAPI(title="PinkFlag", docs_url="/api/docs")
    detector = detector or load_detector()
    usage: dict[tuple[str, str], int] = {}  # (visitor, day) -> messages sent to Sisi

    @app.middleware("http")
    async def visitor(request: Request, call_next):
        """Each browser sends a random anonymous ID; every guy and chat belongs to one ID,
        so visitors never see each other's data. Locally (no ID) everything is 'local'."""
        # header for API calls; ?u= for <img> tags, which can't send headers
        raw = request.headers.get("x-pinkflag-user") or request.query_params.get("u", "")
        owner = raw if VISITOR_ID.fullmatch(raw) else "local"
        token = CURRENT_OWNER.set(owner)
        try:
            # fair-use limit on everything that costs a Gemini / Whisper call
            if daily_limit and request.method == "POST" and request.url.path.endswith(
                    ("/chat", "/voice", "/screenshot", "/transcribe")):
                key = (owner, time.strftime("%Y-%m-%d"))
                if usage.get(key, 0) >= daily_limit:
                    return JSONResponse(status_code=429, content={
                        "reply": f"we've talked a lot today bestie 💕 you've used your {daily_limit} "
                                 "messages for today. come back tomorrow!",
                        "receipts": [], "detail": "daily limit reached"})
                usage[key] = usage.get(key, 0) + 1
            return await call_next(request)
        finally:
            CURRENT_OWNER.reset(token)

    @app.get("/api/config")
    def config():
        """Hosted free version: voice (Whisper + Piper) is off, the UI hides those buttons."""
        return {"voice": os.environ.get("PINKFLAG_VOICE", "1") != "0"}

    @app.delete("/api/me", status_code=200)
    def delete_my_data():
        """Privacy: wipe every guy, chat, note and screenshot of this visitor."""
        return {"deleted_guys": memory.delete_owner(CURRENT_OWNER.get())}

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
        guy = memory.guy(guy_id, owner=CURRENT_OWNER.get())  # other visitors' guys = not found
        if guy is None:
            raise HTTPException(404, "guy not found")
        return guy

    def with_summary(guy: dict) -> dict:
        return {**guy, **summary(memory.analyses(guy["id"]))}

    @app.get("/api/guys")
    def list_guys():
        return [with_summary(g) for g in memory.guys(CURRENT_OWNER.get())]

    @app.post("/api/guys", status_code=201)
    def add_guy(body: NewGuy):
        gid = memory.add_guy(body.name, body.status, body.emoji, owner=CURRENT_OWNER.get())
        return with_summary(get_guy(gid))

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
        return memory.chat_history(f"guy-{guy_id}")

    @app.get("/api/guys/{guy_id}/images/{message_id}")
    def message_image(guy_id: int, message_id: int):
        get_guy(guy_id)
        data = memory.message_image(f"guy-{guy_id}", message_id)
        if not data:
            raise HTTPException(404, "Screenshot unavailable")
        mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
        return Response(data, media_type=mime, headers={"Cache-Control": "no-store"})

    @app.post("/api/guys/{guy_id}/chat")
    def chat(guy_id: int, body: ChatIn):
        return chat_reply(guy_id, body.text)

    @app.post("/api/guys/{guy_id}/screenshot")
    def screenshot(guy_id: int, image: UploadFile, text: str = Form("", max_length=4000)):
        get_guy(guy_id)
        limit = 8 * 1024 * 1024
        data = image.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, "Choose a screenshot under 8 MB.")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            raise HTTPException(415, "Choose a PNG or JPEG screenshot.")
        return chat_reply(guy_id, "[Screenshot] " + text.strip(),
                          {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")})

    def chat_reply(guy_id: int, text: str, image: dict | None = None):
        get_guy(guy_id)
        tools = Toolbox(memory, detector=detector, guy_id=guy_id)
        sisi = Sisi(transport_factory(), memory, tools, session=f"guy-{guy_id}",
                    history_turns=history_turns)
        try:
            reply = sisi.reply(text, image=image)
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

    @app.patch("/api/guys/{guy_id}/receipts/{receipt_id}")
    def correct_rating(guy_id: int, receipt_id: int, body: RatingUpdate):
        if body.label not in LABELS and body.label != "needs_context":
            raise HTTPException(422, "Unknown rating")
        updated = memory.revise_analysis(guy_id, receipt_id, body.label)
        if updated is None:
            raise HTTPException(404, "Receipt not found")
        return _receipt(updated)

    @app.get("/api/guys/{guy_id}/receipts")
    def receipts(guy_id: int):
        guy = with_summary(get_guy(guy_id))
        return {"guy": guy, "timeline": [_receipt(a) for a in memory.analyses(guy_id)],
                "rating_options": {**{k: v.display for k, v in LABELS.items()},
                                   "needs_context": "Needs context"},
                "notes": memory.notes(f"guy-{guy_id}")}

    @app.get("/api/roster/best")
    def best():
        scored = [g for g in (with_summary(g) for g in memory.guys(CURRENT_OWNER.get()))
                  if g["score"] is not None]
        if not scored:
            raise HTTPException(404, "no receipts yet: tell Sisi what they said first")
        return min(scored, key=lambda g: g["score"])

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app
