"""Loading and checking datasets.

Every dataset row has: text, label, source.
`source` matters: it lets us measure whether the model works on *real* messages
or only on the synthetic ones it was trained on.
"""
from pathlib import Path

import pandas as pd

from pinkflag.taxonomy import validate

REQUIRED_COLUMNS = ["text", "label", "source"]


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
