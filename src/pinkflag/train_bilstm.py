"""A BiLSTM text classifier built from scratch in PyTorch (no pre-training).

    python -m pinkflag.train_bilstm --config configs/bilstm.yaml

The middle step of the model comparison:
  classical ML (TF-IDF + LogReg)  ->  deep learning from scratch (this)  ->  transfer learning (XLM-R)

Architecture: tokens -> embedding (learned) -> 2-layer bidirectional LSTM -> max-pool over time
-> dropout -> linear layer with 8 outputs. Everything is learned from our ~600 examples only.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from pinkflag.data import CLEAN_QUALITY, drop_quality
from pinkflag.taxonomy import LABELS
from pinkflag.train_transformer import score, set_seed

LABEL_LIST = list(LABELS)
LABEL2ID = {lab: i for i, lab in enumerate(LABEL_LIST)}
TOKEN = re.compile(r"[a-z0-9']+|[^\sa-z0-9]", re.IGNORECASE)
PAD, UNK = 0, 1


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


class Vocab:
    def __init__(self, texts, min_freq: int = 1, max_size: int = 20000):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:max_len]
        return ids or [UNK]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_labels: int, emb: int = 128, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_labels)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        lengths = (ids != PAD).sum(dim=1).clamp(min=1)
        # packing: each direction of the LSTM only reads real tokens, never the padding
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(ids), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=float("-inf"))
        h = h.max(dim=1).values                           # max-pool over real tokens
        return self.out(self.dropout(h))


def batches(df: pd.DataFrame, vocab: Vocab, max_len: int, size: int, shuffle: bool, seed: int):
    order = np.random.default_rng(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), size):
        rows = df.iloc[order[i:i + size]]
        seqs = [vocab.encode(t, max_len) for t in rows.text]
        ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
        yield ids, torch.tensor([LABEL2ID[lab] for lab in rows.label])


@torch.no_grad()
def predict_ids(model, df, vocab, max_len, device, loss_fn=None):
    model.eval()
    preds, total = [], 0.0
    for ids, y in batches(df, vocab, max_len, 64, False, 0):
        logits = model(ids.to(device))
        if loss_fn is not None:
            total += loss_fn(logits, y.to(device)).item() * len(y)
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds, total / len(df)) if loss_fn is not None else preds


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train a BiLSTM from scratch")
    parser.add_argument("--config", default="configs/bilstm.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t, d = cfg["train"], cfg["data"]
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = (pd.read_csv(d[k]) for k in ("train", "val", "test"))
    train, val = (drop_quality(x, d.get("exclude_label_quality")) for x in (train, val))
    vocab = Vocab(train.text, min_freq=cfg["vocab_min_freq"])
    model = BiLSTMClassifier(len(vocab.itos), len(LABEL_LIST), **cfg["model"]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 BiLSTM from scratch: {params / 1e6:.2f}M parameters · vocab {len(vocab.itos)} · "
          f"{len(train)} training rows · {device}")

    present = np.array(sorted(train.label.unique(), key=LABEL2ID.get))
    w = np.ones(len(LABEL_LIST))
    w[[LABEL2ID[c] for c in present]] = compute_class_weight("balanced", classes=present, y=train.label)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_loss, bad, history = -1.0, float("inf"), 0, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        start, running, n = time.time(), 0.0, 0
        for ids, y in batches(train, vocab, cfg["max_len"], t["batch_size"], True, cfg["seed"] + epoch):
            optimizer.zero_grad()
            loss = loss_fn(model(ids.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running, n = running + loss.item() * len(y), n + len(y)
        val_ids, val_loss = predict_ids(model, val, vocab, cfg["max_len"], device, loss_fn)
        vs = score(val.label, np.array(LABEL_LIST)[val_ids])
        history.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss,
                        "type_macro_f1": vs["type_macro_f1"]})
        print(f"  epoch {epoch:2d}  train loss {running / n:.3f}  val loss {val_loss:.3f}  "
              f"val macro-F1 {vs['type_macro_f1']:.2f}  ({time.time() - start:.1f}s)")
        if vs["type_macro_f1"] > best_f1:
            best_f1 = vs["type_macro_f1"]
            torch.save(model.state_dict(), out_dir / "model.pt")
        if val_loss < best_loss:
            best_loss, bad = val_loss, 0
        else:
            bad += 1
            if bad >= t["early_stopping_patience"]:
                print("  ⏹️  early stopping")
                break

    (out_dir / "vocab.json").write_text(json.dumps(vocab.itos), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps({**cfg["model"], "max_len": cfg["max_len"]}),
                                         encoding="utf-8")
    from pinkflag.train_transformer import plot_history
    plot_history([{**h, "type_accuracy": 0} for h in history],
                 Path(cfg["output"]["report_path"]).with_suffix(".png"))

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    test = test.assign(pred=np.array(LABEL_LIST)[predict_ids(model, test, vocab, cfg["max_len"], device)])
    clean = test[test.label_quality.isin(CLEAN_QUALITY)]
    metrics = {"history": history, "test": score(test.label, test.pred.values),
               "test_clean": score(clean.label, clean.pred.values), "parameters": params}
    c = metrics["test_clean"]
    print(f"\n🧼 CLEAN TEST  acc {c['type_accuracy']:.0%}  macro-F1 {c['type_macro_f1']:.2f}  "
          f"red-flag acc {c['red_flag_accuracy']:.0%}")
    Path(cfg["output"]["report_path"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_bilstm(model_dir: str | Path = "models/bilstm"):
    """Returns predict(texts) -> labels, for evaluation."""
    model_dir = Path(model_dir)
    itos = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    conf = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    max_len = conf.pop("max_len")
    vocab = Vocab([])
    vocab.itos, vocab.stoi = itos, {w: i for i, w in enumerate(itos)}
    model = BiLSTMClassifier(len(itos), len(LABEL_LIST), **conf)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    df_label = LABEL_LIST[0]

    def predict(texts):
        df = pd.DataFrame({"text": list(texts), "label": df_label})
        return np.array(LABEL_LIST)[predict_ids(model, df, vocab, max_len, torch.device("cpu"))]
    return predict


if __name__ == "__main__":
    main()
