"""Run the PinkFlag web app locally.

    python -m pinkflag.app            # then open http://127.0.0.1:8000
"""
import argparse
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
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    key = get_key("GEMINI_API_KEY")
    app = create_app(Memory(cfg["memory_path"]), lambda: gemini_transport(key, cfg["models"]),
                     history_turns=cfg["history_turns"])
    print(f"🎀 PinkFlag running at http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
