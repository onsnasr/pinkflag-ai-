"""Run the PinkFlag web app locally.

    python -m pinkflag.app            # then open http://127.0.0.1:8000
"""
import argparse
import os
from pathlib import Path

import uvicorn
import yaml

from pinkflag.agent.memory import Memory
from pinkflag.agent.sisi import gemini_transport
from pinkflag.app.server import create_app
from pinkflag.env import get_key


def main(argv=None) -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the PinkFlag web app")
    parser.add_argument("--config", default="configs/agent.yaml")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--daily-limit", type=int,
                        default=int(os.environ.get("PINKFLAG_DAILY_LIMIT", "0")) or None,
                        help="max messages per visitor per day (public hosting)")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    key = get_key("GEMINI_API_KEY")
    app = create_app(Memory(cfg["memory_path"]), lambda: gemini_transport(key, cfg["models"]),
                     history_turns=cfg["history_turns"], daily_limit=args.daily_limit)
    # load Whisper + Piper in the background so the first call is instant
    import threading

    def warm():
        try:
            from pinkflag.voice import warm_up
            warm_up()
            print("📞 voice ready (Whisper + Piper loaded)", flush=True)
        except Exception as err:  # noqa: BLE001 - voice extras not installed: chat still works
            print(f"(voice not available: {err})", flush=True)

    threading.Thread(target=warm, daemon=True).start()
    print(f"🎀 PinkFlag running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
