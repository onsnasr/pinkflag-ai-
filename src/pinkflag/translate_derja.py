"""Translate English examples into Tunisian Derja with Gemini (data augmentation).

    python -m pinkflag.translate_derja --config configs/translate.yaml

The label stays the same: "where are you? who are you with?" is controlling in any
language. Output has an empty `review` column: a native speaker marks each row
ok / fix / bad, and only `ok` rows are used for training.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

from pinkflag.datasets.tunisian import script_of
from pinkflag.env import get_key

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """You translate short dating text messages into Tunisian Derja, the way young
Tunisians really text each other.

Style rules:
- Write mostly in Arabizi (Latin letters with numbers: 3=ع 7=ح 9=ق 5=خ 8=غ 2=ء).
- Code-switching is normal: mix in French or English words where a Tunisian would.
- Casual texting: lowercase, little punctuation, "hhh", repeated letters, emojis are fine.
- Spelling varies (tw / taw, barcha / barsha): vary it naturally.
- Keep the SAME meaning and the same emotional intent (controlling stays controlling,
  sweet stays sweet). Don't soften or exaggerate it.
- No real names.

Give {variants} different natural versions for each message.
Return ONLY JSON: a list of objects {{"id": <id>, "derja": ["...", "..."]}}.

Messages:
{messages}
"""


def build_prompt(batch: pd.DataFrame, variants: int) -> str:
    lines = "\n".join(f"{i}: {t}" for i, t in zip(batch.index, batch.text))
    return PROMPT.format(variants=variants, messages=lines)


def parse_response(payload: dict) -> list[dict]:
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def call_gemini(session, key: str, model: str, prompt: str, temperature: float) -> list[dict]:
    resp = session.post(
        API.format(model=model),
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": temperature,
                                   "responseMimeType": "application/json"}},
        timeout=120)
    resp.raise_for_status()
    return parse_response(resp.json())


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Translate examples into Derja")
    parser.add_argument("--config", default="configs/translate.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    key = get_key("GEMINI_API_KEY")
    session = requests.Session()

    src = pd.read_csv(cfg["input"])
    rows = []
    for start in range(0, len(src), cfg["batch_size"]):
        batch = src.iloc[start:start + cfg["batch_size"]]
        results = call_gemini(session, key, cfg["model"],
                              build_prompt(batch, cfg["variants"]), cfg["temperature"])
        for r in results:
            idx = int(r["id"])
            if idx not in batch.index:
                continue
            for derja in r.get("derja", []):
                rows.append({"text": derja.strip(), "label": batch.at[idx, "label"],
                             "text_en": batch.at[idx, "text"], "script": script_of(derja),
                             "source": "gemini_translated", "review": ""})
        print(f"  🌍 {min(start + cfg['batch_size'], len(src))}/{len(src)} translated")

    out = pd.DataFrame(rows).drop_duplicates("text")
    Path(cfg["output"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["output"], index=False)
    print(f"\n✅ {len(out)} Derja examples -> {cfg['output']}")
    print("👀 next: review them (review column: ok / fix / bad)")


if __name__ == "__main__":
    main()
