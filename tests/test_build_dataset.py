import pandas as pd

from pinkflag.build_dataset import balance, split


def _toy(n_groups=60):
    labels = ["green_flag", "gaslighting", "controlling"]
    rows = []
    for g in range(n_groups):
        label = labels[g % 3]
        # every group has an English row + a "translation" sharing the group
        rows.append({"text": f"msg {g}", "label": label, "source": "seed", "group": f"g{g}"})
        rows.append({"text": f"msg {g} derja", "label": label, "source": "mt", "group": f"g{g}"})
    return pd.DataFrame(rows)


def test_split_never_leaks_groups():
    parts = split(_toy(), val=0.15, test=0.15, seed=0)
    train, val, test = (set(parts[k].group) for k in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert sum(len(p) for p in parts.values()) == 120


def test_balance_caps_big_sources():
    df = _toy()
    out = balance(df, {"mt": 5}, seed=0)
    assert out[out.source == "mt"].groupby("label").size().max() == 5
    assert len(out[out.source == "seed"]) == 60
