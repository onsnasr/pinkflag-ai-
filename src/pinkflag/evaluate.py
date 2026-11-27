"""Compare every model on the SAME clean test set, with 95% bootstrap confidence intervals.

    python -m pinkflag.evaluate

Why bootstrap? The clean test set is small (~100 rows), so one accuracy number hides how
uncertain it is. We resample the test set 2,000 times (with replacement), recompute the
metric each time, and report the middle 95% range. Overlapping intervals = no real difference.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pinkflag.data import CLEAN_QUALITY
from pinkflag.taxonomy import GREEN

MODELS = {
    "TF-IDF + LogReg (classical ML)": ("baseline", "models/baseline_strong.joblib"),
    "BiLSTM from scratch (deep learning)": ("bilstm", "models/bilstm"),
    "XLM-RoBERTa fine-tuned (transfer learning)": ("transformer", "models/transformer_strong"),
}


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"accuracy": float((y == p).mean()),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "red_flag_acc": float(((y != GREEN) == (p != GREEN)).mean())}


def bootstrap(y, p, n: int = 2000, seed: int = 0) -> dict:
    """Point estimate + 95% CI for each metric."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    samples = [metrics(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    point = metrics(y, p)
    return {k: {"value": point[k], "low": float(np.percentile([s[k] for s in samples], 2.5)),
                "high": float(np.percentile([s[k] for s in samples], 97.5))} for k in point}


def fmt(m: dict, pct: bool = True) -> str:
    if pct:
        return f"{m['value']:.0%} ({m['low']:.0%}-{m['high']:.0%})"
    return f"{m['value']:.2f} ({m['low']:.2f}-{m['high']:.2f})"


def predictor(kind: str, path: str):
    if kind == "baseline":
        model = joblib.load(path)
        return model.predict
    if kind == "bilstm":
        from pinkflag.train_bilstm import load_bilstm
        return load_bilstm(path)
    from pinkflag.error_analysis import predict_transformer
    return lambda texts: predict_transformer(texts, path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    test = pd.read_csv("data/processed/test.csv")
    clean = test[test.label_quality.isin(CLEAN_QUALITY)].reset_index(drop=True)
    results, lines = {}, [
        f"# Model comparison (clean test set, n = {len(clean)}, 95% bootstrap CI)", "",
        "| Model | Accuracy | Macro-F1 | Red flag vs green |", "|---|---|---|---|"]
    for name, (kind, path) in MODELS.items():
        if not Path(path).exists():
            print(f"  ⏭️  {name}: {path} not found (train it first)")
            continue
        r = bootstrap(clean.label, predictor(kind, path)(clean.text.values))
        results[name] = r
        lines.append(f"| {name} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'], False)} | "
                     f"{fmt(r['red_flag_acc'])} |")
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
