import pandas as pd

from pinkflag.datasets.mentalmanip import convert, map_techniques
from pinkflag.taxonomy import LABELS


def test_every_mapped_label_exists_in_taxonomy():
    from pinkflag.datasets.mentalmanip import NOT_MANIPULATIVE_LABEL, TECHNIQUE_MAP
    targets = {lab for lab, _ in TECHNIQUE_MAP.values() if lab} | {NOT_MANIPULATIVE_LABEL}
    assert targets <= set(LABELS)


def test_map_techniques():
    assert map_techniques("Denial, Feigning Innocence") == (["gaslighting"], "strong")
    assert map_techniques("Denial, Evasion") == (["gaslighting"], "loose")
    assert map_techniques("Intimidation, Shaming or Belittlement") == (
        ["intimidation", "disrespect"], "strong")
    assert map_techniques("Accusation") == ([], "loose")
    assert map_techniques(float("nan")) == ([], "none")


def test_convert():
    raw = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "dialogue": ["Person1: hi\nPerson2: hey", "Person1: that never happened",
                     "Person1: you're worthless\n Person2: stop\nPerson1: or else",
                     "Person1: it's your fault"],
        "manipulative": [0, 1, 1, 1],
        "technique": [None, "Denial", "Shaming or Belittlement, Intimidation", "Accusation"],
    })
    out = convert(raw)
    assert list(out.conversation_id) == ["mm_1", "mm_2", "mm_3"]  # Accusation-only dropped
    assert list(out.label[:2]) == ["green_flag", "gaslighting"]
    assert pd.isna(out.label[2]) and out.labels[2] == "disrespect|intimidation"
    assert out.text[2] == "Person1: you're worthless\nPerson2: stop\nPerson1: or else"
