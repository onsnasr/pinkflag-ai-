"""End-to-end: a real browser drives the real web app (fake Gemini, tiny real detector)."""
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import yaml

from pinkflag.agent.memory import Memory
from pinkflag.app.server import create_app
from pinkflag.data import load_messages
from pinkflag.models.baseline import build_pipeline

sync_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]
REPLY = "bestie that is textbook guilt tripping"


def fake_gemini():
    """First turn: Sisi calls the detector. Second turn: she answers."""
    def factory():
        turns = iter([
            {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {
                "name": "analyze_messages",
                "args": {"messages": ["if you loved me you would cancel your plans"]}}}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": REPLY}]}}]},
        ])
        return lambda body: next(turns)
    return factory


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    df = load_messages(ROOT / cfg["data"]["path"])
    detector = build_pipeline(cfg).fit(df.text, df.label), "test-model"
    memory = Memory(tmp_path_factory.mktemp("e2e") / "memory.db")
    app = create_app(memory, fake_gemini(), detector=detector)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True


@pytest.fixture()
def page(server):
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except sync_api.Error as e:  # browser binaries not installed
            pytest.skip(f"no browser: {e}")
        page = browser.new_page()
        csp = []
        page.on("console", lambda m: "Content Security Policy" in m.text and csp.append(m.text))
        yield page
        assert not csp, f"the CSP blocked something: {csp}"
        browser.close()


def test_add_guy_chat_and_delete(server, page):
    page.goto(server)
    page.click("#privacyOk")
    page.fill("#firstGuyForm input[name=name]", "Youssef")
    page.click("#firstGuyForm button[type=submit]")
    sync_api.expect(page.locator("#tabs")).to_contain_text("Youssef")

    page.fill("#msg", "he said if you loved me you would cancel your plans")
    page.click("#send")
    sync_api.expect(page.locator("#thread")).to_contain_text(REPLY, timeout=15000)
    sync_api.expect(page.locator("#statAnalysed")).to_have_text("1")

    page.on("dialog", lambda d: d.accept())
    page.click("#deleteGuy")
    sync_api.expect(page.locator("#empty")).to_be_visible()


def test_visitors_are_isolated(server, page):
    page.goto(server)
    page.click("#privacyOk")
    sync_api.expect(page.locator("#empty")).to_be_visible()  # a new browser sees nobody
