"""Make the PinkFlag demo video automatically.

    python tools/demo_video.py            # -> _drafts/pinkflag_demo.mp4

1. Gemini text-to-speech reads the narration, one clip per scene.
2. A local PinkFlag server starts on a separate demo database (your real data is untouched).
3. Playwright drives the app in a real browser and records it, each scene lasting as long as
   its narration, with captions drawn on the page.
4. ffmpeg lays the voice clips on top of the recording at the right moments.

Needs: pip install playwright imageio-ffmpeg && python -m playwright install chromium
"""
import base64
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

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
VOICE = "Aoede"  # warm, breezy female voice
STYLE = "Read this warmly and confidently, like a friendly tech presenter, at a natural pace: "

SCENES = [
    ("hook", "Is he husband material, or a walking red flag? Meet PinkFlag: your AI big sister for safer dating."),
    ("chat", ("Paste what he said, and Sisi, our AI agent, gets you the receipts. A classifier we trained "
             "detects the manipulation pattern, and Sisi explains it like a big sister would.")),
    ("receipts", ("Every guy gets his own case file: a vibe meter, a timeline of flags, and a profile Sisi "
                 "fills in as you talk, so you never mix them up.")),
    ("roster", ("Talking to more than one? The roster compares them side by side, and the wheel always "
               "lands on the one who treats you best.")),
    ("call", "You can even call her. Speech recognition and her voice run locally, on the GPU."),
    ("ml", ("Under the hood: four data sources, a leak-free split, and three model families. TF-IDF, a "
           "BiLSTM trained from scratch, and a fine-tuned multilingual transformer, compared with "
           "confidence intervals.")),
    ("end", "PinkFlag. Because every girl deserves someone in her corner."),
]


# ---------- 1. voiceover ----------
def tts(text: str, key: str, path: Path) -> float:
    """Gemini TTS -> WAV file. Returns the duration in seconds."""
    if path.exists():
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    body = {"contents": [{"parts": [{"text": STYLE + text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}}}}
    for attempt in range(6):
        for model in TTS_MODELS:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              headers={"x-goog-api-key": key}, json=body, timeout=120)
            if r.ok:
                pcm = base64.b64decode(r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
                with wave.open(str(path), "wb") as w:  # Gemini returns raw 24 kHz 16-bit mono PCM
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(pcm)
                return len(pcm) / 2 / 24000
            print(f"   tts {model}: {r.status_code}, retrying", flush=True)
        time.sleep(25 * (attempt + 1))  # free tier: a few requests per minute
    raise RuntimeError("Gemini TTS failed")


# ---------- 2. demo server ----------
def start_server(db: Path, key: str) -> None:
    import uvicorn

    from pinkflag import demo_data
    from pinkflag.app.server import create_app

    if db.exists():
        db.unlink()
    demo_data.Memory = lambda: Memory(db)  # seed the demo guys into the demo database only
    demo_data.main()
    memory = Memory(db)
    memory.add_guy("Amine", "talking stage", "🧢")
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
    requests.get(URL + "/api/voice/greeting", timeout=180)  # warm up Piper before recording


# ---------- 3. recording ----------
CAPTION_JS = """(text) => {
  let c = document.getElementById('demo-caption');
  if (!c) {
    c = document.createElement('div'); c.id = 'demo-caption';
    c.style.cssText = 'position:fixed;left:50%;top:78px;transform:translateX(-50%);max-width:80%;pointer-events:none;' +
      'background:rgba(90,35,64,.92);color:#fff;font:600 20px Inter,Segoe UI,sans-serif;padding:12px 22px;' +
      'border-radius:16px;text-align:center;z-index:99;box-shadow:0 10px 30px rgba(0,0,0,.25)';
    document.body.appendChild(c);
  }
  c.textContent = text;
}"""

RESULTS_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>body{margin:0;height:100vh;display:grid;place-items:center;background:#FFF5F8;font-family:Inter,sans-serif;color:#2B1A22}
.card{width:1060px}h1{font-family:Fraunces,serif;color:#5A2340;font-size:44px;margin:0 0 6px}
.sub{color:#8E3B5E;font-size:20px;margin:0 0 22px}.flow{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px}
.flow span{background:#fff;border:1px solid #F2D9E2;border-radius:999px;padding:8px 14px;font-weight:600;color:#5A2340}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden;font-size:19px}
th{background:#8E3B5E;color:#fff;text-align:left;padding:12px 16px}td{padding:12px 16px;border-top:1px solid #F2D9E2}
tr.best td{background:#FBEFF3;font-weight:600}</style></head><body><div class="card">
<h1>Under the hood</h1><p class="sub">Same trusted training data · same clean test set · 95% bootstrap CI</p>
<div class="flow"><span>scraping + MentalManip + seed + Derja</span><span>clean + map labels</span>
<span>leak-free group split</span><span>3 model families</span><span>error analysis</span><span>agent + voice</span></div>
<table><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Red flag vs green</th></tr>
<tr><td>TF-IDF + Logistic Regression</td><td>61% (52-69)</td><td>0.56</td><td>97%</td></tr>
<tr><td>BiLSTM from scratch (PyTorch)</td><td>39% (30-47)</td><td>0.35</td><td>93%</td></tr>
<tr class="best"><td>XLM-RoBERTa fine-tuned (GPU)</td><td>63% (54-72)</td><td>0.58</td><td>94%</td></tr>
</table></div></body></html>"""

END_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>body{margin:0;height:100vh;display:grid;place-items:center;text-align:center;font-family:Inter,sans-serif;
background:radial-gradient(circle at 50% 40%,#8E3B5E,#5A2340 70%);color:#fff}h1{font-family:Fraunces,serif;font-size:72px;margin:0}
p{font-size:24px;opacity:.9}code{background:rgba(255,255,255,.15);padding:6px 14px;border-radius:10px;font-size:20px}</style>
</head><body><div><h1>PinkFlag</h1><p>your AI big sister for safer dating</p><code>github.com/onsnasr/pinkflag-ai-</code></div></body></html>"""


def record(durations: dict[str, float]) -> tuple[Path, dict[str, float]]:
    from playwright.sync_api import sync_playwright

    (OUT / "results.html").write_text(RESULTS_HTML, encoding="utf-8")
    (OUT / "end.html").write_text(END_HTML, encoding="utf-8")
    starts = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1280, "height": 720},
                                  record_video_dir=str(OUT), record_video_size={"width": 1280, "height": 720})
        t0 = time.monotonic()
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_selector("#tabs button")

        def scene(name: str, actions) -> None:
            starts[name] = time.monotonic() - t0
            page.evaluate(CAPTION_JS, dict(SCENES)[name])
            actions()
            left = starts[name] + durations[name] + 0.6 - (time.monotonic() - t0)
            if left > 0:
                page.wait_for_timeout(left * 1000)

        def hook():
            page.click("#openRoster")
            page.wait_for_timeout(1500)
            page.click("#closeRoster")

        def chat():
            page.click("#tabs button:has-text('Amine')")
            page.wait_for_timeout(800)
            page.click("#msg")
            page.keyboard.type('he texted me "where are you rn? send me your location" while I was out '
                               "with my friends 🙄", delay=35)
            page.keyboard.press("Enter")
            page.wait_for_selector(".row.sisi .flag", timeout=90000)
            page.wait_for_timeout(1500)

        def receipts():
            page.click("#tabs button:has-text('Mehdi')")
            page.wait_for_timeout(1200)
            page.mouse.wheel(0, 0)

        def roster():
            page.click("#openRoster")
            page.wait_for_timeout(1200)
            page.locator("#spin").scroll_into_view_if_needed()
            page.click("#spin")
            page.wait_for_timeout(5200)

        def call():
            page.click("#closeRoster")
            page.click("#tabs button:has-text('Karim')")
            page.click("#callBtn")
            page.wait_for_timeout(6000)

        def ml():
            page.click("#hang")
            page.goto((OUT / "results.html").as_uri())
            page.evaluate(CAPTION_JS, dict(SCENES)["ml"])

        def end():
            page.goto((OUT / "end.html").as_uri())
            page.evaluate(CAPTION_JS, dict(SCENES)["end"])

        for name, fn in [("hook", hook), ("chat", chat), ("receipts", receipts), ("roster", roster),
                         ("call", call), ("ml", ml), ("end", end)]:
            print(f"  🎬 {name}", flush=True)
            scene(name, fn)
        page.wait_for_timeout(800)
        video = Path(page.video.path())
        ctx.close()
        browser.close()
    return video, starts


# ---------- 4. edit ----------
def assemble(video: Path, starts: dict[str, float], clips: dict[str, Path], out: Path) -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    inputs, filters = ["-i", str(video)], []
    for i, (name, clip) in enumerate(clips.items(), start=1):
        inputs += ["-i", str(clip)]
        ms = int(starts[name] * 1000)
        filters.append(f"[{i}:a]adelay={ms}|{ms},apad[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(clips) + 1))
    filters.append(f"{mix}amix=inputs={len(clips)}:normalize=0:duration=longest[aout]")
    cmd = [ffmpeg, "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
           "-shortest", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    key = get_key("GEMINI_API_KEY", ROOT / ".env")
    print("🎙️ voiceover (Gemini TTS)…", flush=True)
    clips, durations = {}, {}
    for name, text in SCENES:
        clips[name] = OUT / f"{name}.wav"
        durations[name] = tts(text, key, clips[name])
        print(f"   {name}: {durations[name]:.1f}s", flush=True)
    print("🖥️ starting the demo server…", flush=True)
    start_server(OUT / "demo.db", key)
    print("🎬 recording…", flush=True)
    video, starts = record(durations)
    (OUT / "timeline.json").write_text(json.dumps(starts, indent=2), encoding="utf-8")
    final = ROOT / "_drafts" / "pinkflag_demo.mp4"
    print("✂️ editing…", flush=True)
    assemble(video, starts, clips, final)
    print(f"✅ {final}", flush=True)


if __name__ == "__main__":
    main()
