"""Train + evaluate a model from a config file.

    python -m pinkflag.train --config configs/baseline.yaml

Steps: load data -> cross-validate (honest scores) -> fit on everything ->
save model + metrics -> log the run to MLflow if it's installed.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pinkflag.data import load_messages
from pinkflag.models.baseline import build_pipeline, top_features
from pinkflag.taxonomy import GREEN


def evaluate(cfg: dict, df) -> dict:
    cv = StratifiedKFold(n_splits=cfg["data"]["cv_folds"], shuffle=True,
                         random_state=cfg["seed"])
    proba = cross_val_predict(build_pipeline(cfg), df.text, df.label, cv=cv,
                              method="predict_proba")
    classes = np.array(sorted(df.label.unique()))
    pred = classes[proba.argmax(axis=1)]

    is_red = (df.label != GREEN).astype(int)
    # Verdict = the most likely label is a red-flag type (same rule as Verdict.is_red_flag)
    red_pred = (pred != GREEN).astype(int)

    return {
        "n_messages": len(df),
        "type": {
            "accuracy": float((pred == df.label).mean()),
            "macro_f1": float(f1_score(df.label, pred, average="macro")),
            "report": classification_report(df.label, pred, output_dict=True, zero_division=0),
        },
        "red_flag": {
            "accuracy": float((red_pred == is_red).mean()),
            "macro_f1": float(f1_score(is_red, red_pred, average="macro")),
            "report": classification_report(is_red, red_pred, target_names=["green", "red"],
                                            output_dict=True, zero_division=0),
        },
    }


def print_summary(metrics: dict, learned: dict) -> None:
    t, r = metrics["type"], metrics["red_flag"]
    print(f"\n📊 {metrics['n_messages']} messages, cross-validated\n")
    print(f"🚩 red flag vs green:  accuracy {r['accuracy']:.0%}   macro-F1 {r['macro_f1']:.2f}")
    for name in ["green", "red"]:
        row = r["report"][name]
        print(f"     {name:6s} precision {row['precision']:.2f}  recall {row['recall']:.2f}")
    print(f"\n🏷️  which type:         accuracy {t['accuracy']:.0%}   macro-F1 {t['macro_f1']:.2f}")
    for label in sorted(learned):
        row = t["report"][label]
        print(f"     {label:15s} F1 {row['f1-score']:.2f}   learned: {', '.join(learned[label][:6])}")
    print()


def score(model, df) -> dict:
    """Accuracy + macro-F1 for the type and for red-flag-or-not."""
    pred = model.predict(df.text)
    red_true, red_pred = df.label != GREEN, pred != GREEN
    return {"n": len(df),
            "type_accuracy": float((pred == df.label).mean()),
            "type_macro_f1": float(f1_score(df.label, pred, average="macro", zero_division=0)),
            "red_flag_accuracy": float((red_true == red_pred).mean()),
            "per_label_f1": dict(zip(sorted(df.label.unique()), f1_score(
                df.label, pred, average=None, labels=sorted(df.label.unique()),
                zero_division=0).round(2).tolist()))}


def train_on_splits(cfg: dict):
    d = cfg["data"]
    train, val, test = (pd.read_csv(d[k]) for k in ("train", "val", "test"))
    model = build_pipeline(cfg).fit(train.text, train.label)

    metrics = {"train_rows": len(train), "val": score(model, val), "test": score(model, test)}
    # slices: where does the model do well / badly?
    metrics["test_by_source"] = {s: score(model, g) for s, g in test.groupby("source")}
    metrics["test_by_lang"] = {s: score(model, g) for s, g in test.groupby("lang")}
    derja_gold = Path(d.get("test_derja", ""))
    if derja_gold.is_file() and derja_gold.stat().st_size:
        gold = pd.read_csv(derja_gold)
        if len(gold):
            metrics["test_derja_gold"] = score(model, gold)

    print(f"\n📊 trained on {len(train)} rows")
    for name in ["val", "test"]:
        m = metrics[name]
        print(f"   {name:5s} type acc {m['type_accuracy']:.0%}  macro-F1 {m['type_macro_f1']:.2f}"
              f"  | red-flag acc {m['red_flag_accuracy']:.0%}")
    print("\n🔍 test, sliced:")
    for group in ["test_by_source", "test_by_lang"]:
        for k, m in metrics[group].items():
            print(f"   {k:14s} n={m['n']:4d}  type acc {m['type_accuracy']:.0%}"
                  f"  red-flag acc {m['red_flag_accuracy']:.0%}")
    if "test_derja_gold" in metrics:
        m = metrics["test_derja_gold"]
        print(f"   {'derja (gold)':14s} n={m['n']:4d}  type acc {m['type_accuracy']:.0%}")
    print("\n🏷️  test F1 per label:", metrics["test"]["per_label_f1"], "\n")
    return model, metrics


def log_mlflow(cfg: dict, metrics: dict, model_path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        print("(mlflow not installed: skipping experiment tracking. "
              "pip install -e .[tracking] to enable)")
        return
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run():
        mlflow.log_params({"C": cfg["model"]["C"],
                           "word_ngrams": str(cfg["features"]["word_ngrams"]),
                           "char_ngrams": str(cfg["features"]["char_ngrams"])})
        flat = {}
        for section, values in metrics.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    if isinstance(v, (int, float)):
                        flat[f"{section}_{k}"] = v
        mlflow.log_metrics(flat)
        mlflow.log_artifact(str(model_path))
    print("✅ run logged to MLflow (view with: mlflow ui)")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train a PinkFlag model")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if "train" in cfg["data"]:  # merged dataset with fixed train / val / test splits
        model, metrics = train_on_splits(cfg)
    else:                       # small dataset: cross-validation
        df = load_messages(cfg["data"]["path"])
        metrics = evaluate(cfg, df)
        model = build_pipeline(cfg).fit(df.text, df.label)
        print_summary(metrics, top_features(model))

    model_path = Path(cfg["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path = Path(cfg["output"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"💾 model -> {model_path}   metrics -> {report_path}")

    log_mlflow(cfg, metrics, model_path)


if __name__ == "__main__":
    main()
