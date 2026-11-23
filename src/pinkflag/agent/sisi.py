"""The agent loop: how Sisi thinks, calls tools, and answers.

One turn:
  1. build the request: personality + memory notes + recent conversation + your message
  2. send it to Gemini together with the tool declarations
  3. if Gemini answers with a function call -> run the tool -> send the result back -> repeat
  4. when Gemini answers with text -> that's Sisi's reply; save it to memory
"""
import time
from collections.abc import Callable

import requests

from pinkflag.agent.memory import Memory
from pinkflag.agent.persona import build_system_prompt
from pinkflag.agent.tools import DECLARATIONS, Toolbox

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_TOOL_STEPS = 6

Transport = Callable[[dict], dict]  # request body -> response JSON (swappable for tests)


RETRY_STATUS = {429, 500, 502, 503, 504}  # busy / overloaded: worth retrying


def gemini_transport(key: str, model: str | list[str], timeout: int = 120,
                     retries: int = 3) -> Transport:
    """Send a request to Gemini, retrying when Google is busy and falling back to the
    next model in the list if one stays unavailable."""
    session = requests.Session()
    models = [model] if isinstance(model, str) else list(model)

    def send(body: dict) -> dict:
        last_err = None
        for name in models:
            for attempt in range(retries):
                try:
                    resp = session.post(API.format(model=name), headers={"x-goog-api-key": key},
                                        json=body, timeout=timeout)
                except (requests.Timeout, requests.ConnectionError) as err:
                    last_err = err
                else:
                    if resp.status_code not in RETRY_STATUS:
                        resp.raise_for_status()
                        return resp.json()
                    last_err = requests.HTTPError(f"{resp.status_code} from {name}")
                time.sleep(2 * (attempt + 1))
        raise last_err

    return send


class Sisi:
    def __init__(self, transport: Transport, memory: Memory, tools: Toolbox,
                 session: str = "default", history_turns: int = 30,
                 on_tool: Callable[[str, dict, dict], None] | None = None):
        self.send = transport
        self.memory = memory
        self.tools = tools
        self.session = session
        self.history_turns = history_turns
        self.on_tool = on_tool  # hook to show tool calls (for learning/debugging)
        self.tools.scope = session  # notes are remembered per session (= per guy in the app)

    def _memory_notes(self) -> list[str]:
        notes = list(self.memory.notes(self.session))
        guy = self.memory.guy(self.tools.guy_id) if self.tools.guy_id is not None else None
        if guy:
            notes.insert(0, f"This chat is about {guy['name']} (status: {guy['status']}).")
            labels = {"works_or_studies": "works/studies", "met_at": "where they met",
                      "age": "age", "detail": "detail"}
            for field, label in labels.items():
                if guy.get(field):
                    notes.insert(1, f"His profile: {label}: {guy[field]}")
        return notes

    def _contents(self, user_text: str) -> list[dict]:
        contents = [{"role": "user" if role == "user" else "model", "parts": [{"text": text}]}
                    for role, text in self.memory.history(self.session, self.history_turns)]
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        return contents

    def reply(self, user_text: str, voice: bool = False) -> str:
        contents = self._contents(user_text)
        for _ in range(MAX_TOOL_STEPS):
            # rebuilt every step: facts she just remembered are visible right away
            system = build_system_prompt(self._memory_notes(), voice=voice)
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "tools": [{"functionDeclarations": DECLARATIONS}],
                "generationConfig": {"temperature": 0.8},
            }
            data = self.send(body)
            candidate = (data.get("candidates") or [{}])[0]
            content = candidate.get("content") or {"role": "model", "parts": []}
            parts = content.get("parts", [])
            calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if not calls:
                text = "".join(p.get("text", "") for p in parts).strip()
                text = text or "sorry bestie, my brain glitched 😵 can you say that again?"
                self.memory.add_message(self.session, "user", user_text)
                self.memory.add_message(self.session, "model", text)
                return text

            # keep Gemini's turn exactly as sent (incl. any thought signatures), then answer it
            contents.append(content)
            responses = []
            for call in calls:
                result = self.tools.call(call["name"], call.get("args", {}))
                if self.on_tool:
                    self.on_tool(call["name"], call.get("args", {}), result)
                responses.append({"functionResponse": {"name": call["name"],
                                                       "response": result}})
            contents.append({"role": "user", "parts": responses})

        return "okay I went down a rabbit hole there 🙈 can you ask me again?"
