"""Merge every source into one dataset and split it into train / val / test.

    python -m pinkflag.build_dataset --config configs/dataset.yaml

1. harmonize   every source -> text, label, source, lang, unit, label_quality, group
2. balance     cap big sources so they don't drown the small ones
3. split       by *group*, so related rows never sit on both sides of the split
               (a message and its Derja translations share a group)
4. save        data/processed/{train,val,test,test_derja}.csv + stats.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from pinkflag.taxonomy import validate

COLUMNS = ["text", "label", "source", "lang", "unit", "label_quality", "group"]


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"  ⏭️  missing, skipped: {path}")
        return pd.DataFrame()
    return pd.read_csv(p)


def harmonize(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = cfg["sources"]
    parts = []

    hw = _read(s["handwritten"])
    if len(hw):
        parts.append(pd.DataFrame({
            "text": hw.text, "label": hw.label, "source": hw.source, "lang": "en",
            "unit": "message", "label_quality": "gold",
            "group": [f"seed_{i}" for i in hw.index]}))

    sc = _read(s["scraped"])
    if len(sc):
        parts.append(pd.DataFrame({
            "text": sc.text, "label": sc.label, "source": "scraped", "lang": "en",
            "unit": "message", "label_quality": "weak_" + sc.label_method,
            "group": [f"scraped_{i}" for i in sc.index]}))

    mm = _read(s["mentalmanip"])
    if len(mm):
        mm = mm[mm.label.notna()]  # single-label dialogues only, for now
        parts.append(pd.DataFrame({
            "text": mm.text, "label": mm.label, "source": "mentalmanip", "lang": "en",
            "unit": "dialogue", "label_quality": "dataset_" + mm.map_quality,
            "group": mm.conversation_id}))

    mt = _read(s["derja_mt"])
    if len(mt):
        review = mt.review.astype("string").fillna("").str.strip().str.lower()
        keep = review.isin(["ok", "fix"]) | (
            (review == "") & cfg.get("use_unreviewed_translations", False))
        mt = mt[keep]
        parts.append(pd.DataFrame({
            "text": mt.text, "label": mt.label, "source": "derja_mt", "lang": "derja",
            "unit": "message",
            "label_quality": ["translated_reviewed" if r else "translated_unreviewed"
                              for r in review[keep]],
            "group": [f"seed_{i}" for i in mt.src_id]}))  # same group as its English original

    gold = _read(s["derja_gold"])
    derja_test = pd.DataFrame({
        "text": gold.get("text", []), "label": gold.get("label", []),
        "source": "derja_gold", "lang": "derja", "unit": "message",
        "label_quality": "gold", "group": [f"derja_{i}" for i in gold.index]})

    df = pd.concat(parts, ignore_index=True)[COLUMNS]
    df["text"] = df.text.astype(str).str.strip()
    df = df[df.text != ""].drop_duplicates("text")
    validate(df.label)
    if len(derja_test):
        validate(derja_test.label)
    return df.reset_index(drop=True), derja_test


def balance(df: pd.DataFrame, caps: dict, seed: int) -> pd.DataFrame:
    out = []
    for (source, _label), grp in df.groupby(["source", "label"]):
        cap = caps.get(source)
        out.append(grp.sample(cap, random_state=seed) if cap and len(grp) > cap else grp)
    return pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, val: float, test: float, seed: int) -> dict[str, pd.DataFrame]:
    """Group-aware stratified split: carve out test, then val from what's left."""
    def carve(frame, frac):
        k = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        rest_idx, part_idx = next(sgkf.split(frame, frame.label, frame.group))
        return frame.iloc[rest_idx], frame.iloc[part_idx]

    rest, test_df = carve(df, test)
    train_df, val_df = carve(rest.reset_index(drop=True), val / (1 - test))
    return {"train": train_df, "val": val_df, "test": test_df}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build the PinkFlag dataset")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    df, derja_test = harmonize(cfg)
    print(f"🧩 harmonized: {len(df)} rows")
    df = balance(df, cfg.get("cap_per_source_label", {}), cfg["seed"])
    print(f"⚖️  balanced:   {len(df)} rows")
    splits = split(df, cfg["split"]["val"], cfg["split"]["test"], cfg["seed"])
    splits["test_derja"] = derja_test

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, frame in splits.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        stats[name] = {"rows": len(frame),
                       "by_label": frame.label.value_counts().to_dict() if len(frame) else {},
                       "by_source": frame.source.value_counts().to_dict() if len(frame) else {}}
        print(f"💾 {name:10s} {len(frame):5d} rows")
    leaks = set(splits["train"].group) & (set(splits["val"].group) | set(splits["test"].group))
    stats["group_leaks"] = len(leaks)
    print(f"🔒 groups shared between train and val/test: {len(leaks)}")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nlabels in train:")
    print(splits["train"].label.value_counts().to_string())


if __name__ == "__main__":
    main()
