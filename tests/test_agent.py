"""Sisi's loop tested with a fake Gemini: no key, no internet."""
import copy

from pinkflag.agent.memory import Memory
from pinkflag.agent.persona import build_system_prompt
from pinkflag.agent.sisi import Sisi
from pinkflag.agent.tools import DECLARATIONS, Toolbox


class FakeTools(Toolbox):
    def __init__(self, memory, guy_id=None):
        self.memory, self.model, self.model_name = memory, None, None
        self.guy_id, self.scope, self.analysed = guy_id, "default", []

    def analyze_messages(self, messages):
        return {"results": [{"message": m, "label": "controlling", "confidence": 0.9}
                            for m in messages]}


def fake_gemini(script):
    """Returns canned responses in order and records every request body."""
    calls = []

    def send(body):
        calls.append(copy.deepcopy(body))  # the loop keeps extending the same list
        return script[len(calls) - 1]
    return send, calls


def text(t):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t}]}}]}


def function_call(name, args):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": name, "args": args}}]}}]}


def test_plain_reply_is_saved_to_history():
    mem = Memory(":memory:")
    send, calls = fake_gemini([text("heyy 💕")])
    sisi = Sisi(send, mem, FakeTools(mem))
    assert sisi.reply("hi sisi") == "heyy 💕"
    assert mem.history("default") == [("user", "hi sisi"), ("model", "heyy 💕")]
    assert calls[0]["tools"][0]["functionDeclarations"] == DECLARATIONS


def test_tool_call_round_trip():
    mem = Memory(":memory:")
    send, calls = fake_gemini([
        function_call("analyze_messages", {"messages": ["send me ur location rn"]}),
        function_call("remember", {"note": "he asks for her location"}),
        text("that's controlling bestie ⛓️"),
    ])
    seen = []
    sisi = Sisi(send, mem, FakeTools(mem), on_tool=lambda n, a, r: seen.append(n))
    assert sisi.reply("he said: send me ur location rn") == "that's controlling bestie ⛓️"
    assert seen == ["analyze_messages", "remember"]
    assert mem.notes() == ["he asks for her location"]
    # 2nd request carries Gemini's function call + our function response
    sent = calls[1]["contents"]
    assert "functionCall" in sent[-2]["parts"][0]
    assert sent[-1]["parts"][0]["functionResponse"]["response"]["results"][0]["label"] == (
        "controlling")
    # notes are injected into the next system prompt
    assert "he asks for her location" in calls[2]["systemInstruction"]["parts"][0]["text"]


def test_unknown_tool_is_handled():
    mem = Memory(":memory:")
    assert "error" in FakeTools(mem).call("delete_everything", {})


def test_system_prompt_with_and_without_memory():
    assert "fresh start" in build_system_prompt([])
    assert "- his name is Amine" in build_system_prompt(["his name is Amine"])
