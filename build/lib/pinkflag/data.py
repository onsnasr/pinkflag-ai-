"""Loading and checking datasets.

Every dataset row has: text, label, source.
`source` matters: it lets us measure whether the model works on *real* messages
or only on the synthetic ones it was trained on.
"""
from pathlib import Path

import pandas as pd

from pinkflag.taxonomy import validate

REQUIRED_COLUMNS = ["text", "label", "source"]


# Labels we trust most (see docs/LABEL_MAPPING.md): hand-written, strong dataset mappings,
# and weak labels where heading + keywords agreed. Used for the "clean test" score.
CLEAN_QUALITY = {"gold", "dataset_strong", "weak_rule_agree", "translated_reviewed"}


def drop_quality(df: pd.DataFrame, exclude: list[str] | None) -> pd.DataFrame:
    """Remove rows whose label_quality is in `exclude` (e.g. loose mappings)."""
    if not exclude or "label_quality" not in df:
        return df
    return df[~df.label_quality.isin(exclude)].reset_index(drop=True)


def load_messages(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    dupes = df.duplicated("text").sum()
    if dupes:
        # duplicates leak between train and test folds and inflate scores
        df = df.drop_duplicates("text")

    validate(df["label"])
    return df.reset_index(drop=True)
