import pandas as pd

from pinkflag.datasets.tunisian import build, clean, script_of


def test_script_detection():
    assert script_of("3lech ma tjawebch") == "latin"
    assert script_of("علاش ما تجاوبش") == "arabic"
    assert script_of("bravo يعطيك الصحة") == "mixed"


def test_clean_removes_private_bits():
    assert clean("ya @salma3 call +216 22 333 444 https://x.co/a") == (
        "ya <user> call <phone> <url>")


def test_build_dedupes_and_drops_tiny():
    frames = [pd.DataFrame({"text": ["rabi m3ak", "rabi  m3ak", "ok"],
                            "sentiment": ["positive"] * 3, "source": ["tsac"] * 3})]
    out = build(frames)
    assert list(out.text) == ["rabi m3ak"]
    assert out.script[0] == "latin"
