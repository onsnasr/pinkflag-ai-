"""Fine-tune a multilingual transformer (XLM-RoBERTa) for PinkFlag.

    python -m pinkflag.train_transformer --config configs/transformer.yaml

Transfer learning in 4 steps:
  1. load a model pre-trained on 100 languages (it already "knows" English, Arabic, French)
  2. put a fresh classification head on top: 8 outputs, one per PinkFlag label
  3. train everything for a few epochs on our data, with a small learning rate
  4. keep the epoch with the best validation macro-F1, then report on the test set

Written as a plain PyTorch loop (no Trainer magic) so every step is visible.
Runs on a GPU if there is one (CUDA), otherwise on CPU (slow).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from pinkflag.taxonomy import GREEN, LABELS

LABEL_LIST = list(LABELS)
LABEL2ID = {lab: i for i, lab in enumerate(LABEL_LIST)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, max_length, batch_size, shuffle):
    enc = tokenizer(list(df.text), truncation=True, max_length=max_length)
    items = [{"input_ids": ids, "attention_mask": mask, "labels": LABEL2ID[lab]}
             for ids, mask, lab in zip(enc["input_ids"], enc["attention_mask"], df.label)]

    def collate(batch):  # pad each batch only to its longest text: faster than padding to max
        padded = tokenizer.pad([{k: b[k] for k in ("input_ids", "attention_mask")}
                                for b in batch], return_tensors="pt")
        padded["labels"] = torch.tensor([b["labels"] for b in batch])
        return padded

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device, loss_fn=None):
    """Predicted label ids; with loss_fn, also the average loss (used for validation loss)."""
    model.eval()
    preds, total, n = [], 0.0, 0
    for batch in loader:
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        if loss_fn is not None:
            total += loss_fn(logits.float(), labels).item() * len(labels)
            n += len(labels)
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds, total / max(n, 1)) if loss_fn is not None else preds


def freeze_bottom(model, embeddings: bool, n_layers: int) -> str:
    base = getattr(model, model.base_model_prefix)  # e.g. model.roberta
    parts = []
    if embeddings:
        for p in base.embeddings.parameters():
            p.requires_grad = False
        parts.append("embeddings")
    for layer in base.encoder.layer[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    if n_layers:
        parts.append(f"bottom {n_layers} layers")
    return " + ".join(parts) or "nothing"


def plot_history(history: list[dict], path: Path) -> None:
    """Train vs validation loss + validation macro-F1 per epoch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: skipping the loss chart)")
        return
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(ep, [h["train_loss"] for h in history], "o-", color="#8E3B5E", label="train loss")
    a.plot(ep, [h["val_loss"] for h in history], "o-", color="#E88FAB", label="validation loss")
    a.set_xlabel("epoch"); a.set_title("Loss (lower = better)"); a.legend()
    b.plot(ep, [h["type_macro_f1"] for h in history], "o-", color="#3E8E6E")
    b.set_xlabel("epoch"); b.set_title("Validation macro-F1 (higher = better)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"📈 loss curves -> {path}")


def score(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    labels = sorted(y_true.unique())
    return {"n": len(y_true),
            "type_accuracy": float((y_pred == y_true.values).mean()),
            "type_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "red_flag_accuracy": float(((y_true.values != GREEN) == (y_pred != GREEN)).mean()),
            "per_label_f1": dict(zip(labels, f1_score(
                y_true, y_pred, labels=labels, average=None, zero_division=0).round(2).tolist()))}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Fine-tune a transformer")
    parser.add_argument("--config", default="configs/transformer.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t = cfg["train"]
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    train, val, test = (pd.read_csv(cfg["data"][k]) for k in ("train", "val", "test"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABEL_LIST),
        id2label=dict(enumerate(LABEL_LIST)), label2id=LABEL2ID).to(device)

    train_dl = make_loader(train, tokenizer, cfg["max_length"], t["batch_size"], shuffle=True)
    val_dl = make_loader(val, tokenizer, cfg["max_length"], t["batch_size"] * 2, shuffle=False)
    test_dl = make_loader(test, tokenizer, cfg["max_length"], t["batch_size"] * 2, shuffle=False)

    weights = None
    if t.get("class_weights") == "balanced":
        present = np.array(sorted(train.label.unique(), key=LABEL2ID.get))
        w = compute_class_weight("balanced", classes=present, y=train.label)
        full = np.ones(len(LABEL_LIST))
        full[[LABEL2ID[c] for c in present]] = w
        weights = torch.tensor(full, dtype=torch.float, device=device)
    # label smoothing: aim for 90% sure instead of 100%, so the model can't over-memorize noisy labels
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights,
                                        label_smoothing=t.get("label_smoothing", 0.0))
    eval_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)  # plain loss, comparable across runs

    # freezing: keep the embeddings + bottom layers as pre-trained, only adapt the top layers
    frozen = freeze_bottom(model, t.get("freeze_embeddings", False), t.get("freeze_layers", 0))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧊 frozen: {frozen} · training {trainable / 1e6:.0f}M of {total / 1e6:.0f}M weights")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=t["lr"], weight_decay=t["weight_decay"])
    total_steps = len(train_dl) * t["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(t["warmup_ratio"] * total_steps), total_steps)
    use_amp = t.get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg["output"]["model_dir"])
    best_f1, best_val_loss, bad_epochs, history = -1.0, float("inf"), 0, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 10 == 0 or step == len(train_dl):
                print(f"\r  epoch {epoch}  step {step}/{len(train_dl)}  loss {running / step:.3f}",
                      end="", flush=True)

        val_ids, val_loss = predict(model, val_dl, device, eval_loss_fn)
        val_score = score(val.label, np.array(LABEL_LIST)[val_ids])
        train_loss = running / len(train_dl)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{
            k: val_score[k] for k in ("type_accuracy", "type_macro_f1", "red_flag_accuracy")}})
        # train loss falling while val loss rises = overfitting (memorizing, not learning)
        gap = "⚠️ overfitting?" if len(history) > 1 and val_loss > history[-2]["val_loss"] \
            and train_loss < history[-2]["train_loss"] else ""
        print(f"\r  epoch {epoch}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  "
              f"val acc {val_score['type_accuracy']:.0%}  macro-F1 {val_score['type_macro_f1']:.2f}"
              f"  ({time.time() - start:.0f}s) {gap}")

        # keep the checkpoint with the best validation macro-F1 (the metric we report)...
        if val_score["type_macro_f1"] > best_f1:
            best_f1 = val_score["type_macro_f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"     💾 best so far (epoch {epoch})")
        # ...but stop when validation LOSS stops improving: the earliest overfitting signal
        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= t["early_stopping_patience"]:
                print(f"  ⏹️  early stopping: val loss hasn't improved for {bad_epochs} epochs")
                break

    plot_history(history, Path(cfg["output"]["report_path"]).with_suffix(".png"))

    # reload the best epoch and evaluate once on the test set
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    test_pred = np.array(LABEL_LIST)[predict(model, test_dl, device)]
    metrics = {"model": cfg["model_name"], "history": history, "best_val_macro_f1": best_f1,
               "test": score(test.label, test_pred)}
    test = test.assign(pred=test_pred)
    for col in ("source", "lang"):
        metrics[f"test_by_{col}"] = {k: score(g.label, g.pred.values) for k, g in test.groupby(col)}

    m = metrics["test"]
    print(f"\n📊 TEST  type acc {m['type_accuracy']:.0%}  macro-F1 {m['type_macro_f1']:.2f}"
          f"  | red-flag acc {m['red_flag_accuracy']:.0%}")
    for col in ("test_by_source", "test_by_lang"):
        for k, s in metrics[col].items():
            print(f"   {k:20s} n={s['n']:4d}  type acc {s['type_accuracy']:.0%}"
                  f"  red-flag acc {s['red_flag_accuracy']:.0%}")
    print("🏷️  test F1 per label:", m["per_label_f1"])

    report = Path(cfg["output"]["report_path"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"💾 model -> {out_dir}   metrics -> {report}")


if __name__ == "__main__":
    main()
