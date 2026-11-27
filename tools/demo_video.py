"""Make the PinkFlag demo video automatically.

    python tools/demo_video.py            # -> _drafts/pinkflag_demo.mp4 (+ _fast.mp4)

1. Gemini text-to-speech: a narrator voice, plus a second voice playing "the user" on the call.
2. A local PinkFlag server starts on a separate demo database (your real data is untouched).
3. Playwright drives the app in a real browser and records it. The "user" voice is fed into a
   fake microphone, so the call is real: Whisper hears it, Sisi answers, Piper speaks.
4. Every sound (narration, ringtone, greeting, user, Sisi) is placed on the timeline with ffmpeg.

Needs: pip install playwright imageio-ffmpeg && python -m playwright install chromium-headless-shell
"""
import base64
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinkflag.agent.memory import Memory
from pinkflag.agent.sisi import gemini_transport
from pinkflag.env import get_key

OUT = ROOT / "_drafts" / "demo_video"
PORT = 8010
URL = f"http://127.0.0.1:{PORT}"
TTS_MODELS = ["gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview"]
NARRATOR, USER_VOICE = "Aoede", "Kore"
STYLE = "Read this warmly and confidently, like a friendly tech presenter, at a natural pace: "
USER_STYLE = "Say this like a young woman talking to her best friend on the phone, a bit upset: "
DEMO_VISITOR = "demo-visitor-0000-0000-000000000001"
USER_LINE = "Hey Sisi. He just told me I'm overreacting, again. Is that normal?"

SCENES = [
    ("hook", ("A fine-tuned multilingual transformer. A BiLSTM trained from scratch. An LLM agent with "
              "tools and memory, and speech models running on a laptop GPU. All built to answer one "
              "question: is he husband material, or a walking red flag? Meet PinkFlag.")),
    ("chat", ("Paste what he said. Sisi, our AI agent, runs a classifier we trained, and remembers the "
              "whole conversation, so patterns don't slip by.")),
    ("screenshot", "Or just drop a screenshot. Vision reads who said what, and the receipts follow."),
    ("call", "You can even call her. Speech recognition and her voice run locally, on the GPU."),
    ("roster", ("Talking to more than one? The roster compares them side by side, and the wheel always "
                "lands on the one who treats you best.")),
    ("ml", ("Under the hood: four data sources, a leak-free split, and three model families, compared "
            "with confidence intervals. And every push runs unit, end-to-end and security scans.")),
    ("end", "PinkFlag is live, and free. Because every girl deserves someone in her corner."),
]


# ---------- audio helpers ----------
def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def write_wav(path: Path, pcm: bytes, rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def tts(text: str, key: str, path: Path, voice: str = NARRATOR, style: str = STYLE) -> float:
    """Gemini TTS -> WAV (cached). Returns the duration in seconds."""
    if path.exists():
        return wav_seconds(path)
    body = {"contents": [{"parts": [{"text": style + text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    for attempt in range(6):
        for model in TTS_MODELS:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": key}, json=body, timeout=120)
            if r.ok:
                pcm = base64.b64decode(r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
                write_wav(path, pcm)  # Gemini returns raw 24 kHz 16-bit mono PCM
                return len(pcm) / 2 / 24000
            print(f"   tts {model}: {r.status_code}, retrying", flush=True)
        time.sleep(25 * (attempt + 1))  # free tier: a few requests per minute
    raise RuntimeError("Gemini TTS failed")


def ringtone(path: Path, seconds: float) -> None:
    """Two-tone phone ring (440 + 480 Hz, 1 s on / 1 s off), like the one in the app."""
    rate = 24000
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.12 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t))
    tone *= ((t % 2.0) < 1.0)
    write_wav(path, (tone * 32767).astype(np.int16).tobytes(), rate)


def save_b64_wav(b64: str, path: Path) -> float:
    path.write_bytes(base64.b64decode(b64))
    return wav_seconds(path)


# ---------- demo server ----------
def start_server(db: Path, key: str) -> None:
    import uvicorn

    from pinkflag import demo_data
    from pinkflag.app.server import create_app
    from pinkflag.voice import whisper_model

    if db.exists():
        db.unlink()
    demo_data.Memory = lambda: Memory(db)  # seed the demo guys into the demo database only
    demo_data.main()
    memory = Memory(db)
    memory.db.execute("UPDATE guys SET owner = ?", (DEMO_VISITOR,))  # visitors only see their own guys
    memory.db.commit()
    memory.add_guy("Amine", "talking stage", "🧢", owner=DEMO_VISITOR)
    app = create_app(memory, lambda: gemini_transport(key, ["gemini-flash-lite-latest",
                                                            "gemini-3.1-flash-lite"]))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):
        try:
            requests.get(URL + "/api/guys", timeout=2)
            break
        except requests.RequestException:
            time.sleep(0.5)
    requests.get(URL + "/api/voice/greeting", timeout=180)  # load Piper before recording
    whisper_model()                                          # load Whisper before recording


# ---------- pages ----------
CAPTION_JS = """([text, bottom]) => {
  let c = document.getElementById('demo-caption');
  if (!c) {
    c = document.createElement('div'); c.id = 'demo-caption';
    c.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);max-width:80%;pointer-events:none;' +
      'background:rgba(90,35,64,.92);color:#fff;font:600 20px Inter,Segoe UI,sans-serif;padding:12px 22px;' +
      'border-radius:16px;text-align:center;z-index:99;box-shadow:0 10px 30px rgba(0,0,0,.25)';
    document.body.appendChild(c);
  }
  c.style.top = bottom ? 'auto' : '78px'; c.style.bottom = bottom ? '36px' : 'auto';
  c.textContent = text;
}"""

CHAT_SHOT_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;width:390px;background:#fff;font:16px -apple-system,Segoe UI,sans-serif}
.top{background:#f6f6f6;padding:14px;text-align:center;font-weight:600;border-bottom:1px solid #ddd}
.c{padding:14px;display:flex;flex-direction:column;gap:8px}
.m{max-width:75%;padding:9px 13px;border-radius:18px}.l{background:#e9e9eb;align-self:flex-start}
.r{background:#0a84ff;color:#fff;align-self:flex-end}.t{text-align:center;color:#999;font-size:12px}
</style></head><body><div class="top">Amine 🧢</div><div class="c"><div class="t">Today 21:40</div>
<div class="m r">I'm at dinner with my sister 🍝</div>
<div class="m l">why didn't you answer me for 20 min</div>
<div class="m l">you're always with your friends, never with me</div>
<div class="m l">if you really loved me you'd come home now</div>
</div></body></html>"""

RESULTS_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>body{margin:0;height:100vh;display:grid;place-items:start center;padding-top:28px;box-sizing:border-box;background:#FFF5F8;font-family:Inter,sans-serif;color:#2B1A22}
.card{width:1060px}h1{font-family:Fraunces,serif;color:#5A2340;font-size:44px;margin:0 0 6px}
.sub{color:#8E3B5E;font-size:20px;margin:0 0 22px}.flow{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.flow span{background:#fff;border:1px solid #F2D9E2;border-radius:999px;padding:8px 14px;font-weight:600;color:#5A2340}
table{width:100%;border-collapse:collapse;background:#fff;font-size:19px}
th{background:#8E3B5E;color:#fff;text-align:left;padding:12px 16px}td{padding:9px 16px;border-top:1px solid #F2D9E2}
tr.best td{background:#FBEFF3;font-weight:600}</style></head><body><div class="card">
<h1>Under the hood</h1><p class="sub">Same trusted training data · same clean test set · 95% bootstrap CI</p>
<div class="flow"><span>scraping + MentalManip + seed + Derja</span><span>label-noise audit (κ = 0.15)</span>
<span>leak-free group split</span><span>3 model families</span><span>agent + memory + vision + voice</span></div>
<div class="flow"><span>✅ pytest</span><span>✅ Playwright e2e</span><span>✅ Bandit + CodeQL (SAST)</span>
<span>✅ pip-audit (SCA)</span><span>✅ OWASP ZAP (DAST)</span><span>✅ Gitleaks</span></div>
<table><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Red flag vs green</th></tr>
<tr><td>TF-IDF + Logistic Regression</td><td>61% (52-69)</td><td>0.56</td><td>97%</td></tr>
<tr><td>BiLSTM from scratch (PyTorch)</td><td>39% (30-47)</td><td>0.35</td><td>93%</td></tr>
<tr class="best"><td>XLM-RoBERTa fine-tuned (GPU)</td><td>63% (54-72)</td><td>0.58</td><td>94%</td></tr>
</table></div></body></html>"""

END_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>body{margin:0;height:100vh;display:grid;place-items:center;text-align:center;font-family:Inter,sans-serif;
background:radial-gradient(circle at 50% 40%,#8E3B5E,#5A2340 70%);color:#fff}h1{font-family:Fraunces,serif;font-size:72px;margin:0}
p{font-size:24px;opacity:.9}code{display:block;margin:8px auto;width:max-content;background:rgba(255,255,255,.15);
padding:6px 14px;border-radius:10px;font-size:19px}
.live{display:inline-block;margin:6px 0 18px;background:#fff;color:#8E3B5E;font-weight:600;font-size:28px;
padding:12px 26px;border-radius:999px;box-shadow:0 8px 30px rgba(0,0,0,.25)}</style></head><body><div><h1>PinkFlag</h1>
<p>your AI big sister for safer dating</p>
<div class="live">try it live 👉 pinkflag-0lwy.onrender.com</div><code>github.com/onsnasr/pinkflag-ai-</code>
<code>huggingface.co/onsn/pinkflag-detector</code></div></body></html>"""


# ---------- recording ----------
def record(narr: dict[str, float], user_wav: Path) -> tuple[Path, list[tuple[float, Path]]]:
    from playwright.sync_api import sync_playwright

    for name, html in [("results", RESULTS_HTML), ("end", END_HTML), ("chatshot", CHAT_SHOT_HTML)]:
        (OUT / f"{name}.html").write_text(html, encoding="utf-8")
    sounds: list[tuple[float, Path]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={user_wav}"])
        # 1) make the fake chat screenshot (not recorded)
        shot_page = browser.new_page(viewport={"width": 390, "height": 330})
        shot_page.goto((OUT / "chatshot.html").as_uri())
        shot_page.screenshot(path=str(OUT / "chat_screenshot.png"))
        shot_page.close()

        ctx = browser.new_context(bypass_csp=True,  # the recorder injects helper scripts
                                  viewport={"width": 1280, "height": 720}, permissions=["microphone"],
                                  record_video_dir=str(OUT), record_video_size={"width": 1280, "height": 720})
        ctx.add_init_script(f"localStorage.setItem('pinkflag-visitor', '{DEMO_VISITOR}');"
                            "localStorage.setItem('pinkflag-privacy-ok', '1');")
        t0 = time.monotonic()
        now = lambda: time.monotonic() - t0
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_selector("#tabs button")
        if page.locator("#privacy.open").count():
            page.click("#privacyOk")

        def caption(text, bottom=False):
            page.evaluate(CAPTION_JS, [text, bottom])

        def scene(name, actions, bottom=False):
            start = now()
            sounds.append((start, OUT / f"{name}.wav"))
            caption(dict(SCENES)[name], bottom)
            actions()
            left = start + narr[name] + 0.5 - now()
            if left > 0:
                page.wait_for_timeout(left * 1000)

        def sisi_replies():
            return page.locator(".row.sisi").count()

        def send(text):
            before = sisi_replies()
            page.click("#msg")
            page.keyboard.type(text, delay=22)
            page.keyboard.press("Enter")
            page.wait_for_function(f"document.querySelectorAll('.row.sisi .bubble:not(.typing)').length > {before}",
                                   timeout=120000)
            page.wait_for_timeout(900)

        def hook():
            page.click("#openRoster")
            page.wait_for_timeout(1800)
            page.click("#closeRoster")

        def chat():
            page.click("#tabs button:has-text('Amine')")
            page.wait_for_timeout(600)
            send('we\'ve been talking for 2 months. last night he texted "where are you rn? send me your location"')
            send('and today he said "you never told me you were going out, you\'re making things up again"')

        def screenshot():
            before = sisi_replies()
            page.set_input_files("#imagePicker", str(OUT / "chat_screenshot.png"))
            page.wait_for_timeout(900)
            page.click("#msg")
            page.keyboard.type("he sent this tonight 😩", delay=22)
            page.keyboard.press("Enter")
            page.wait_for_function(f"document.querySelectorAll('.row.sisi .bubble:not(.typing)').length > {before}",
                                   timeout=120000)
            page.wait_for_timeout(1200)

        def call():
            page.wait_for_timeout(narr["call"] * 1000)  # narration first, then the call itself
            click = now()
            with page.expect_response("**/api/voice/greeting", timeout=120000) as greet:
                page.click("#callBtn")
            greet_wav = OUT / "greeting_live.wav"
            g_len = save_b64_wav(greet.value.json()["audio"], greet_wav)
            g_start = max(now(), click + 3.0)  # the app rings for at least 3 s
            ringtone(OUT / "ring.wav", g_start - click)
            sounds.extend([(click, OUT / "ring.wav"), (g_start, greet_wav)])
            caption("Sisi: " + greet.value.json()["text"], bottom=True)
            page.wait_for_timeout((g_start - now() + g_len + 0.4) * 1000)
            # "you" talk: the fake microphone plays the user's recorded line
            page.click("#talk")
            sounds.append((now(), user_wav))
            caption("You: " + USER_LINE, bottom=True)
            page.wait_for_timeout((wav_seconds(user_wav) + 0.6) * 1000)
            with page.expect_response("**/voice", timeout=180000) as reply:
                page.click("#talk")
            data = reply.value.json()
            r_wav = OUT / "reply_live.wav"
            r_len = save_b64_wav(data["audio"], r_wav)
            sounds.append((now(), r_wav))
            caption("Sisi: " + data["reply"], bottom=True)
            page.wait_for_timeout((r_len + 0.8) * 1000)
            page.click("#hang")

        def roster():
            page.click("#openRoster")
            page.wait_for_timeout(1000)
            page.locator("#spin").scroll_into_view_if_needed()
            page.click("#spin")
            page.wait_for_timeout(5200)

        def ml():
            page.click("#closeRoster")
            page.goto((OUT / "results.html").as_uri())
            caption(dict(SCENES)["ml"], bottom=True)

        def end():
            page.goto((OUT / "end.html").as_uri())
            caption(dict(SCENES)["end"], bottom=True)

        for name, fn in [("hook", hook), ("chat", chat), ("screenshot", screenshot), ("call", call),
                         ("roster", roster), ("ml", ml), ("end", end)]:
            print(f"  🎬 {name}", flush=True)
            scene(name, fn)
        page.wait_for_timeout(1000)
        video = Path(page.video.path())
        ctx.close()
        browser.close()
    return video, sounds


# ---------- edit ----------
def dead_air(ffmpeg: str, video: Path, longest: float = 1.6, keep: float = 0.5,
             tail: float = 2.5) -> list[tuple[float, float]]:
    """Silent stretches (Sisi thinking, waiting for Gemini) to cut, keeping a short breath
    on each side and a few seconds of the end card."""
    log = subprocess.run([ffmpeg, "-i", str(video), "-af", f"silencedetect=n=-40dB:d={longest}",
                          "-f", "null", "-"], check=False, capture_output=True, text=True).stderr
    starts = [float(x.split("silence_start:")[1]) for x in log.splitlines() if "silence_start:" in x]
    ends = [float(x.split("silence_end:")[1].split("|")[0]) for x in log.splitlines() if "silence_end:" in x]
    cuts = []
    for i, s in enumerate(starts):
        if s < 0.5:  # the opening hook starts silent on purpose
            continue
        if i < len(ends):
            cuts.append((s + keep, ends[i] - keep))
        else:  # silence until the end: keep a little of the end card
            cuts.append((s + tail, 1e9))
    return [(s, e) for s, e in cuts if e > s]


def assemble(video: Path, sounds: list[tuple[float, Path]], out: Path) -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    inputs, filters = ["-i", str(video)], []
    for i, (t, wav) in enumerate(sounds, start=1):
        inputs += ["-i", str(wav)]
        ms = int(t * 1000)
        filters.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms},apad[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(sounds) + 1))
    filters.append(f"{mix}amix=inputs={len(sounds)}:normalize=0:duration=longest[aout]")
    subprocess.run([ffmpeg, "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v",
                    "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", str(out)], check=True, capture_output=True)
    fast = out.with_name(out.stem + "_fast.mp4")
    gaps = dead_air(ffmpeg, out)
    drop = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in gaps) or "0"
    print(f"   cut {sum(e - s for s, e in gaps):.1f}s of waiting", flush=True)
    subprocess.run([ffmpeg, "-y", "-i", str(out), "-filter_complex",
                    (f"[0:v]select='not({drop})',setpts=N/FRAME_RATE/TB,setpts=PTS/1.1[v];"
                    f"[0:a]aselect='not({drop})',asetpts=N/SR/TB,atempo=1.1[a]"),
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                    str(fast)], check=True, capture_output=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    key = get_key("GEMINI_API_KEY", ROOT / ".env")
    print("🎙️ voices (Gemini TTS)…", flush=True)
    narr = {}
    for name, text in SCENES:
        narr[name] = tts(text, key, OUT / f"{name}.wav")
        print(f"   {name}: {narr[name]:.1f}s", flush=True)
    raw_user = OUT / "user_raw.wav"
    tts(USER_LINE, key, raw_user, voice=USER_VOICE, style=USER_STYLE)
    user_wav = OUT / "user.wav"  # 0.4 s of silence first, so the mic catches the whole line
    with wave.open(str(raw_user)) as w:
        pcm = w.readframes(w.getnframes())
    write_wav(user_wav, b"\x00\x00" * int(24000 * 0.4) + pcm)
    print("🖥️ starting the demo server…", flush=True)
    start_server(OUT / "demo.db", key)
    print("🎬 recording…", flush=True)
    video, sounds = record(narr, user_wav)
    (OUT / "timeline.json").write_text(json.dumps([(t, p.name) for t, p in sounds], indent=2),
                                       encoding="utf-8")
    print("✂️ editing…", flush=True)
    final = ROOT / "_drafts" / "pinkflag_demo.mp4"
    assemble(video, sounds, final)
    print(f"✅ {final} (+ pinkflag_demo_fast.mp4)", flush=True)


if __name__ == "__main__":
    main()
