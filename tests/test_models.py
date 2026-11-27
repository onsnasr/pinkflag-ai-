import numpy as np
import torch

from pinkflag.evaluate import bootstrap, metrics
from pinkflag.train_bilstm import PAD, BiLSTMClassifier, Vocab, tokenize


def test_tokenize_keeps_slang_and_emoji():
    assert tokenize("U up?? 🙄") == ["u", "up", "?", "?", "🙄"]


def test_vocab_unknown_words_and_padding_ids():
    v = Vocab(["send me your location"])
    ids = v.encode("send me pizza", max_len=10)
    assert ids[0] == v.stoi["send"] and ids[-1] == 1  # 'pizza' -> <unk>
    assert v.itos[PAD] == "<pad>"


def test_bilstm_forward_shape_ignores_padding():
    model = BiLSTMClassifier(vocab_size=50, n_labels=8, emb=8, hidden=8).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 0, 0]])
    longer = torch.tensor([[5, 6, 7, 0, 0, 0, 0], [5, 6, 7, 0, 0, 0, 0]])
    out = model(ids)
    assert out.shape == (2, 8)
    assert torch.allclose(out, model(longer), atol=1e-6)  # extra padding changes nothing


def test_bootstrap_interval_contains_point_estimate():
    y = np.array(["green_flag", "gaslighting"] * 30)
    p = y.copy()
    p[:10] = "controlling"
    r = bootstrap(y, p, n=300)
    for m in ("accuracy", "macro_f1", "red_flag_acc"):
        assert r[m]["low"] <= r[m]["value"] <= r[m]["high"]
    assert metrics(y, y)["accuracy"] == 1.0
