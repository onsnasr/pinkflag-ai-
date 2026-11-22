"""Translate English examples into Tunisian Derja with Gemini (data augmentation).

    python -m pinkflag.translate_derja --config configs/translate.yaml

The label stays the same: "where are you? who are you with?" is controlling in any
language. Output has an empty `review` column: a native speaker marks each row
ok / fix / bad, and only `ok` rows are used for training.
"""
import argparse
import json
import sys
import time
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


# forces Gemini to answer in exactly this shape: [{"id": 3, "derja": ["...", "..."]}]
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "INTEGER"},
                       "derja": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["id", "derja"],
    },
}


def build_prompt(batch: pd.DataFrame, variants: int) -> str:
    lines = "\n".join(f"{i}: {t}" for i, t in zip(batch.index, batch.text))
    return PROMPT.format(variants=variants, messages=lines)


def parse_response(payload: dict) -> list[dict]:
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def call_gemini(session, key: str, model: str, prompt: str, temperature: float,
                retries: int = 3) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(
                API.format(model=model),
                headers={"x-goog-api-key": key},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature,
                                           "responseMimeType": "application/json",
                                           "responseSchema": RESPONSE_SCHEMA}},
                timeout=300)
            if resp.status_code in (429, 500, 503) and attempt < retries:
                raise requests.HTTPError(resp.status_code)
            resp.raise_for_status()
            return parse_response(resp.json())
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError,
                json.JSONDecodeError, KeyError) as err:
            if attempt == retries:
                print(f"  ⚠️  batch failed after {retries} tries ({err}), skipping")
                return []
            wait = 10 * attempt
            print(f"  ⏳ {type(err).__name__}, retrying in {wait}s...")
            time.sleep(wait)
    return []


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
    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip messages translated in an earlier run (free-tier limits interrupt runs)
    done = pd.read_csv(out_path) if out_path.exists() and out_path.stat().st_size else None
    rows = done.to_dict("records") if done is not None else []
    done_ids = set(done.src_id) if done is not None else set()
    todo = src[~src.index.isin(done_ids)]
    print(f"  {len(done_ids)} already translated, {len(todo)} to go")

    for start in range(0, len(todo), cfg["batch_size"]):
        batch = todo.iloc[start:start + cfg["batch_size"]]
        results = call_gemini(session, key, cfg["model"],
                              build_prompt(batch, cfg["variants"]), cfg["temperature"])
        for r in results:
            idx = int(r.get("id", -1))
            if idx not in batch.index:
                continue
            for derja in r.get("derja", []):
                if not isinstance(derja, str) or not derja.strip():
                    continue
                rows.append({"text": derja.strip(), "label": batch.at[idx, "label"],
                             "text_en": batch.at[idx, "text"], "script": script_of(derja),
                             "source": "gemini_translated", "src_id": idx, "review": ""})
        pd.DataFrame(rows).drop_duplicates("text").to_csv(out_path, index=False)  # save often
        print(f"  🌍 {min(start + cfg['batch_size'], len(todo))}/{len(todo)} "
              f"({len(results)} returned)")
        time.sleep(cfg.get("pause_seconds", 0))

    out = pd.read_csv(out_path) if rows else pd.DataFrame()
    print(f"\n✅ {len(out)} Derja examples -> {out_path}")
    print("👀 next: review them (review column: ok / fix / bad)")


if __name__ == "__main__":
    main()
