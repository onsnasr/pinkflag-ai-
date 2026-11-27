"""Tunisian Derja text: TSAC + TUNIZI -> one cleaned file.

    python -m pinkflag.datasets.tunisian

These datasets are labeled for *sentiment*, not red flags. We use them to teach
the model how Tunisians write (Arabizi like "3lech", Arabic script, French mixing),
e.g. for continued pre-training of a multilingual transformer. Red-flag labels in
Derja come from hand-written + translated examples (docs/DERJA_GUIDE.md).

Sources
  TSAC   github.com/fbougares/TSAC            LGPL-3.0   ~17k Facebook comments
  TUNIZI zenodo.org/records/4275240           CC BY 4.0  Arabizi YouTube comments
"""
import re
import sys
from pathlib import Path

import pandas as pd
import requests

RAW = Path("data/raw/datasets")
OUT = Path("data/interim/tunisian_text.csv")
TSAC_URL = "https://raw.githubusercontent.com/fbougares/TSAC/master/{}.txt"
TUNIZI_URL = "https://zenodo.org/api/records/4275240/files/tunizi_train/content"

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
PHONE_RE = re.compile(r"\+?\d[\d\s]{7,}\d")
SENTIMENT = {1: "positive", 0: "neutral", -1: "negative"}


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def script_of(text: str) -> str:
    a, latin = len(ARABIC.findall(text)), len(LATIN.findall(text))
    if a and latin:
        return "mixed"
    return "arabic" if a else "latin"


def clean(text: str) -> str:
    text = URL_RE.sub("<url>", str(text))
    text = MENTION_RE.sub("<user>", text)  # privacy: no usernames
    text = PHONE_RE.sub("<phone>", text)
    return re.sub(r"\s+", " ", text).strip()


def load_tsac() -> pd.DataFrame:
    frames = []
    for split in ["train", "test"]:
        for pol in ["pos", "neg"]:
            path = _download(TSAC_URL.format(f"{split}_{pol}"), RAW / "tsac" / f"{split}_{pol}.txt")
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            frames.append(pd.DataFrame({
                "text": lines, "sentiment": "positive" if pol == "pos" else "negative"}))
    return pd.concat(frames).assign(source="tsac")


def load_tunizi() -> pd.DataFrame:
    df = pd.read_csv(_download(TUNIZI_URL, RAW / "tunizi" / "tunizi_train"))
    return pd.DataFrame({"text": df.text, "sentiment": df.label.map(SENTIMENT),
                         "source": "tunizi"})


def build(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df.text.map(clean)
    df = df[df.text.str.len() >= 3]
    df = df.drop_duplicates("text")
    df["script"] = df.text.map(script_of)
    return df[["text", "sentiment", "script", "source"]].reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = build([load_tsac(), load_tunizi()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"🇹🇳 Tunisian texts: {len(df)}")
    print(f"   by source:    {df.source.value_counts().to_dict()}")
    print(f"   by script:    {df.script.value_counts().to_dict()}")
    print(f"   by sentiment: {df.sentiment.value_counts().to_dict()}")
    print(f"💾 -> {OUT}")
    print("\nsample:")
    print(df.sample(8, random_state=3).to_string(max_colwidth=60))


if __name__ == "__main__":
    main()
