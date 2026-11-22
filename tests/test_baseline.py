from pathlib import Path

import pytest
import yaml

from pinkflag.data import load_messages
from pinkflag.models.baseline import build_pipeline, predict
from pinkflag.taxonomy import LABELS, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df(cfg):
    return load_messages(ROOT / cfg["data"]["path"])


@pytest.fixture(scope="module")
def model(cfg, df):
    return build_pipeline(cfg).fit(df.text, df.label)


def test_seed_data_uses_known_labels_only(df):
    validate(df.label)
    assert set(df.label) == set(LABELS), "every label needs at least one example"


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        validate(["green_flag", "situationship_energy"])


def test_every_class_has_enough_examples(df):
    assert df.label.value_counts().min() >= 10


def test_prediction_shape(model):
    v = predict(model, ["hey"])[0]
    assert 0.0 <= v.red_flag_prob <= 1.0
    assert len(v.top_labels) == 3
    assert abs(sum(p for _, p in v.top_labels)) <= 1.0 + 1e-9


def test_obvious_cases_smoke(model):
    """Not an accuracy test: just catches a completely broken model."""
    control = predict(model, ["send me ur location rn and who are you with"])[0]
    sweet = predict(model, ["text me when you get home safe, have fun tonight"])[0]
    assert control.red_flag_prob > sweet.red_flag_prob
