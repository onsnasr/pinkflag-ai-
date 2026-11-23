"""Chat with Sisi in the terminal.

    python -m pinkflag.agent                 # continue your chat
    python -m pinkflag.agent --session amine # a separate chat (e.g. per guy)
    python -m pinkflag.agent --show-tools    # see which tools Sisi calls (learning mode)

Commands inside the chat:  /notes  /forget  /quit
"""
import argparse
import json
import sys
from pathlib import Path

import requests
import yaml

from pinkflag.agent.memory import Memory
from pinkflag.agent.sisi import Sisi, gemini_transport
from pinkflag.agent.tools import Toolbox
from pinkflag.env import get_key


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chat with Sisi")
    parser.add_argument("--config", default="configs/agent.yaml")
    parser.add_argument("--session", default="default")
    parser.add_argument("--show-tools", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    memory = Memory(cfg["memory_path"])
    tools = Toolbox(memory)

    def show(name, call_args, result):
        print(f"   🔧 {name}({json.dumps(call_args, ensure_ascii=False)[:120]})")
        print(f"      → {json.dumps(result, ensure_ascii=False)[:300]}")

    sisi = Sisi(gemini_transport(get_key("GEMINI_API_KEY"), cfg["models"]), memory, tools,
                session=args.session, history_turns=cfg["history_turns"],
                on_tool=show if args.show_tools else None)

    print("👧🏽 Sisi: heyy bestie 💕 what's going on? spill.\n"
          "   (/notes = what I remember · /forget = wipe my memory · /quit)\n")
    while True:
        try:
            text = input("💬 you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/notes":
            notes = memory.notes()
            print("\n".join(f"   📝 {n}" for n in notes) or "   (nothing yet)")
            continue
        if text == "/forget":
            memory.forget_everything()
            print("   🧽 memory wiped")
            continue
        try:
            print(f"\n👧🏽 Sisi: {sisi.reply(text)}\n")
        except (requests.RequestException, KeyError, ValueError) as err:
            # network / quota errors shouldn't kill the chat
            print(f"   ⚠️  couldn't reach Sisi's brain ({err}). try again in a bit.\n")
    print("👧🏽 Sisi: byee, text me anytime 💕")


if __name__ == "__main__":
    main()
