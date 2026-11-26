"""The vibe score: one number per guy on the 💍 husband material ↔ walking red flag 🚩 meter.

0 = all green flags, 100 = all red flags. Serious flags weigh more than mild ones.
It describes the messages she shared, never the whole person.
"""
from collections import Counter

from pinkflag.taxonomy import GREEN, LABELS

SEVERITY = {
    "intimidation": 1.6, "controlling": 1.4, "gaslighting": 1.4, "disrespect": 1.2,
    "guilt_tripping": 1.0, "love_bombing": 1.0, "breadcrumbing": 0.7,
}


def vibe_score(analyses: list[dict]) -> int | None:
    if not analyses:
        return None
    red = sum(SEVERITY.get(a["label"], 1.0) for a in analyses if a["label"] in SEVERITY)
    green = sum(1.0 for a in analyses if a["label"] == GREEN)
    return round(100 * red / (red + green)) if red + green else None


def verdict(score: int | None) -> str:
    if score is None:
        return "no receipts yet"
    if score < 35:
        return "green flag energy"
    if score < 65:
        return "mixed signals"
    return "leaning red flag"


def summary(analyses: list[dict]) -> dict:
    counts = Counter(a["label"] for a in analyses)
    top = [{"label": k, "emoji": LABELS[k].emoji, "name": LABELS[k].display, "count": n}
           for k, n in counts.most_common() if k in LABELS]
    score = vibe_score(analyses)
    return {"score": score, "verdict": "needs context" if analyses and score is None else verdict(score), "top_flags": top,
            "red_flags": sum(n for k, n in counts.items() if k in SEVERITY),
            "green_flags": counts.get(GREEN, 0), "needs_context": counts["needs_context"], "analysed": len(analyses)}
