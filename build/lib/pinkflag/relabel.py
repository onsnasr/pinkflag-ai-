"""LLM-as-annotator: a second, independent opinion on loosely mapped labels.

    python -m pinkflag.relabel --config configs/relabel.yaml

MentalManip techniques were mapped to our labels with "loose" matches (e.g. Persuasion -> love
bombing) and "not manipulative" was mapped to green flag. Error analysis showed many of those
labels are wrong. Here Gemini reads each dialogue + our label definitions and picks a label
*without seeing the original one*. Later, build_dataset promotes a row to trusted only when both
annotators agree. Output is resumable (rows already done are skipped).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

from pinkflag.agent.sisi import gemini_transport
from pinkflag.env import get_key
from pinkflag.taxonomy import LABELS

PROMPT = """You are an expert annotator of manipulation in conversations.

Labels (pick exactly ONE per dialogue, judging how Person1 or Person2 treats the other):
{definitions}

Rules:
- Pick the single label that best describes the most notable behaviour in the dialogue.
- Use green_flag only if the dialogue is respectful and free of the other behaviours.
- Judge only what is written. Do not guess hidden context.

Return JSON: a list of {{"id": <id>, "label": "<one of the labels>", "confidence": <0-1>}}.

Dialogues:
{dialogues}
"""

SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {
        "id": {"type": "INTEGER"},
        "label": {"type": "STRING", "enum": list(LABELS)},
        "confidence": {"type": "NUMBER"}},
        "required": ["id", "label", "confidence"]},
}


def definitions() -> str:
    return "\n".join(f"- {k}: {v.definition}" for k, v in LABELS.items())


def build_prompt(batch: pd.DataFrame, max_chars: int) -> str:
    parts = [f"[{i}]\n{str(t)[:max_chars]}" for i, t in zip(batch.index, batch.text)]
    return PROMPT.format(definitions=definitions(), dialogues="\n\n".join(parts))


def parse(payload: dict) -> list[dict]:
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Second-opinion labels with Gemini")
    parser.add_argument("--config", default="configs/relabel.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    src = pd.read_csv(cfg["input"])
    src = src[src.map_quality.isin(cfg["relabel_quality"]) & src.label.notna()]
    out = Path(cfg["output"])
    done = pd.read_csv(out) if out.exists() and out.stat().st_size else pd.DataFrame(
        columns=["conversation_id", "llm_label", "llm_confidence"])
    todo = src[~src.conversation_id.isin(done.conversation_id)]
    print(f"🏷️  {len(src)} dialogues to relabel · {len(done)} already done · {len(todo)} to go")

    send = gemini_transport(get_key("GEMINI_API_KEY"), cfg["models"])
    rows = done.to_dict("records")
    for start in range(0, len(todo), cfg["batch_size"]):
        batch = todo.iloc[start:start + cfg["batch_size"]]
        body = {"contents": [{"parts": [{"text": build_prompt(batch, cfg["max_chars"])}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                                     "responseSchema": SCHEMA}}
        results = None
        for attempt in range(1, 6):  # Google overloads come in waves: back off and retry
            try:
                results = parse(send(body))
                break
            except Exception as err:  # noqa: BLE001 - one bad batch must not stop the run
                wait = 30 * attempt
                print(f"  ⏳ {type(err).__name__}, waiting {wait}s (try {attempt}/5)", flush=True)
                time.sleep(wait)
        if results is None:
            print("  ⚠️  batch skipped (will be retried on the next run)")
            continue
        for r in results:
            if r.get("id") in batch.index and r.get("label") in LABELS:
                rows.append({"conversation_id": batch.at[r["id"], "conversation_id"],
                             "llm_label": r["label"], "llm_confidence": r.get("confidence")})
        pd.DataFrame(rows).drop_duplicates("conversation_id").to_csv(out, index=False)
        print(f"  ✅ {min(start + cfg['batch_size'], len(todo))}/{len(todo)}", flush=True)
        time.sleep(cfg["pause_seconds"])

    res = pd.read_csv(out).merge(src[["conversation_id", "label"]], on="conversation_id")
    agree = (res.llm_label == res.label).mean()
    kappa = cohen_kappa_score(res.label, res.llm_label)
    print(f"\n🤝 agreement with the original mapping: {agree:.0%} · Cohen's kappa {kappa:.2f} "
          f"(n={len(res)})")
    print(pd.crosstab(res.label, res.llm_label).to_string())


if __name__ == "__main__":
    main()
