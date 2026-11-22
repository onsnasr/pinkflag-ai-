"""MentalManip (ACL 2024) -> PinkFlag format.

    python -m pinkflag.datasets.mentalmanip

Source: https://huggingface.co/datasets/audreyeleven/MentalManip  (CC BY-SA 4.0)
We use the *consensus* file: dialogues where all annotators agreed.

Each row is a whole dialogue (not one message), labeled with 0+ techniques.
We translate techniques to PinkFlag labels (see docs/LABEL_MAPPING.md):
  - `labels`  every PinkFlag label the dialogue shows (multi-label)
  - `label`   filled only when there is exactly ONE, so single-label models can use it
"""
import sys
from pathlib import Path

import pandas as pd
import requests

URL = "https://huggingface.co/datasets/audreyeleven/MentalManip/resolve/main/mentalmanip_con.csv"
RAW = Path("data/raw/datasets/mentalmanip/mentalmanip_con.csv")
OUT = Path("data/interim/mentalmanip.csv")

# technique -> (PinkFlag label, how good the match is)
TECHNIQUE_MAP = {
    "Denial": ("gaslighting", "strong"),
    "Feigning Innocence": ("gaslighting", "strong"),
    "Evasion": ("gaslighting", "loose"),
    "Rationalization": ("gaslighting", "loose"),
    "Playing Victim Role": ("guilt_tripping", "strong"),
    "Playing Servant Role": ("guilt_tripping", "loose"),
    "Shaming or Belittlement": ("disrespect", "strong"),
    "Intimidation": ("intimidation", "strong"),
    "Brandishing Anger": ("intimidation", "strong"),
    "Persuasion or Seduction": ("love_bombing", "loose"),
    "Accusation": (None, "unmapped"),  # blame-shifting: no clean PinkFlag label (yet)
}
NOT_MANIPULATIVE_LABEL = "green_flag"


def download(url: str = URL, path: Path = RAW) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def map_techniques(raw: str | float) -> tuple[list[str], str]:
    """'Denial, Accusation' -> (['gaslighting'], 'strong')."""
    if not isinstance(raw, str) or not raw.strip():
        return [], "none"
    labels, strengths = [], set()
    for tech in (t.strip() for t in raw.split(",")):
        label, strength = TECHNIQUE_MAP.get(tech, (None, "unknown"))
        strengths.add(strength)
        if label and label not in labels:
            labels.append(label)
    quality = "strong" if strengths == {"strong"} else "loose"
    return labels, quality


def convert(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.itertuples(index=False):
        text = "\n".join(line.strip() for line in str(rec.dialogue).splitlines() if line.strip())
        if int(rec.manipulative) == 0:
            labels, quality = [NOT_MANIPULATIVE_LABEL], "not_manipulative"
        else:
            labels, quality = map_techniques(rec.technique)
            if not labels:
                continue  # manipulative, but only via techniques we can't map
        rows.append({
            "text": text,
            "label": labels[0] if len(labels) == 1 else None,
            "labels": "|".join(labels),
            "map_quality": quality,
            "unit": "dialogue",
            "source": "mentalmanip",
            "conversation_id": f"mm_{rec.id}",
        })
    out = pd.DataFrame(rows)
    return out.drop_duplicates("text").reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw = pd.read_csv(download())
    out = convert(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    single = out[out.label.notna()]
    print(f"📥 MentalManip dialogues:   {len(raw)}")
    print(f"✅ converted:               {len(out)}  (dropped {len(raw) - len(out)}: unmappable)")
    print(f"🏷️  single-label:            {len(single)}")
    print(single.label.value_counts().to_string())
    print(f"🔀 multi-label:             {(out.label.isna()).sum()}")
    print(f"🎯 match quality:           {out.map_quality.value_counts().to_dict()}")
    print(f"💾 -> {OUT}")


if __name__ == "__main__":
    main()
