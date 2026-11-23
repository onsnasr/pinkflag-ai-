"""Error analysis: where does a model go wrong, and why?

    python -m pinkflag.error_analysis --model baseline
    python -m pinkflag.error_analysis --model transformer

Outputs (in reports/):
  confusion_<model>.png    rows = true label, columns = predicted label
  errors_<model>.csv       every wrong prediction, with its source and label quality
and prints the most common confusions + accuracy by label quality.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from pinkflag.taxonomy import LABELS

LABEL_LIST = list(LABELS)


def predict_baseline(texts, path="models/baseline_merged.joblib"):
    return joblib.load(path).predict(texts)


def predict_transformer(texts, path="models/transformer"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(list(texts[i:i + 32]), truncation=True, max_length=192, padding=True,
                      return_tensors="pt").to(device)
            preds += model(**enc).logits.argmax(-1).tolist()
    return np.array([model.config.id2label[p] for p in preds])


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    pink = LinearSegmentedColormap.from_list("pink", ["#FFF5F8", "#F4B6C8", "#8E3B5E", "#5A2340"])
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)  # each row = 100% of that true label
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(norm, cmap=pink, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("what the model PREDICTED")
    ax.set_ylabel("the TRUE label")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "#2B1A22")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Error analysis on the test set")
    parser.add_argument("--model", choices=["baseline", "transformer"], default="transformer")
    parser.add_argument("--test", default="data/processed/test.csv")
    args = parser.parse_args(argv)

    test = pd.read_csv(args.test)
    fn = predict_baseline if args.model == "baseline" else predict_transformer
    test["pred"] = fn(test.text.values)
    test["correct"] = test.pred == test.label

    labels = [lab for lab in LABEL_LIST if lab in set(test.label) | set(test.pred)]
    cm = confusion_matrix(test.label, test.pred, labels=labels)
    out = Path("reports")
    plot_confusion(cm, labels, f"{args.model}: confusion matrix (test set)",
                   out / f"confusion_{args.model}.png")

    print(f"\n🔍 {args.model}: {test.correct.mean():.0%} correct on {len(test)} test rows\n")
    pairs = [(labels[i], labels[j], cm[i, j]) for i in range(len(labels))
             for j in range(len(labels)) if i != j and cm[i, j]]
    print("Most common mix-ups (true → predicted):")
    for t, p, n in sorted(pairs, key=lambda x: -x[2])[:8]:
        print(f"   {n:3d} × {t:15s} → {p}")

    print("\nAccuracy by label quality (how trustworthy the answer key is):")
    print(test.groupby("label_quality").correct.agg(["mean", "size"])
          .rename(columns={"mean": "accuracy", "size": "rows"})
          .sort_values("rows", ascending=False).to_string(float_format=lambda x: f"{x:.0%}"))

    errors = test[~test.correct][["text", "label", "pred", "source", "label_quality"]]
    errors.to_csv(out / f"errors_{args.model}.csv", index=False)
    print(f"\n💾 {out / f'confusion_{args.model}.png'}   {out / f'errors_{args.model}.csv'}")


if __name__ == "__main__":
    main()
