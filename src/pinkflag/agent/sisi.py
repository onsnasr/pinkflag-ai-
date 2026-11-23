"""The agent loop: how Sisi thinks, calls tools, and answers.

One turn:
  1. build the request: personality + memory notes + recent conversation + your message
  2. send it to Gemini together with the tool declarations
  3. if Gemini answers with a function call -> run the tool -> send the result back -> repeat
  4. when Gemini answers with text -> that's Sisi's reply; save it to memory
"""
from collections.abc import Callable

import requests

from pinkflag.agent.memory import Memory
from pinkflag.agent.persona import build_system_prompt
from pinkflag.agent.tools import DECLARATIONS, Toolbox

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_TOOL_STEPS = 6

Transport = Callable[[dict], dict]  # request body -> response JSON (swappable for tests)


def gemini_transport(key: str, model: str, timeout: int = 120) -> Transport:
    session = requests.Session()

    def send(body: dict) -> dict:
        resp = session.post(API.format(model=model), headers={"x-goog-api-key": key},
                            json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

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

    def _contents(self, user_text: str) -> list[dict]:
        contents = [{"role": "user" if role == "user" else "model", "parts": [{"text": text}]}
                    for role, text in self.memory.history(self.session, self.history_turns)]
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        return contents

    def reply(self, user_text: str) -> str:
        contents = self._contents(user_text)
        for _ in range(MAX_TOOL_STEPS):
            body = {
                "systemInstruction": {"parts": [{"text": build_system_prompt(self.memory.notes())}]},
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
