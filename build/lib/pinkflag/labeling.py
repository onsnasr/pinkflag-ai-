"""Clean scraped quotes and give them weak labels.

    python -m pinkflag.labeling

Step 1  clean       drop noise: advice/replies, article sentences, titles
Step 2  weak label  guess the label from the heading above the quote + keywords
Step 3  review      rows the rules aren't sure about go to a spreadsheet for a human

Every row keeps `label_method` (rule_agree / rule_context / rule_text / human)
so we always know how trustworthy each label is.
"""
import re
import sys
from pathlib import Path

import pandas as pd

# ---- step 1: cleaning rules ---------------------------------------------------

# headings above *replies* or advice, not red-flag messages
NOISE_CONTEXT = re.compile(
    r"respond|response|shut down|protect yourself|how to|resource|boundar|say instead"
    r"|what to say|stop breadcrumbing others|you don.t have to allow", re.IGNORECASE)

# article language about manipulators, not something a partner would text
META_WORDS = re.compile(
    r"gaslight|narciss|breadcrumb|love.?bomb|manipulat|therapist|abuser|victim|"
    r"relationship|partner|someone|people|a boss|co-worker|research|studies", re.IGNORECASE)

# a real message talks to/about "you" or "me"
PRONOUN = re.compile(r"\b(you|your|you're|youre|ur|u|i|i'm|im|me|my|we|us)\b", re.IGNORECASE)


def drop_reason(text: str, context: str) -> str | None:
    """Why this quote isn't a usable message (None = keep it)."""
    text = text.replace("’", "'")
    if NOISE_CONTEXT.search(context or ""):
        return "advice_or_reply"
    if len(text.split()) > 25:
        return "too_long"
    if ":" in text:
        return "title"
    if META_WORDS.search(text):
        return "article_language"
    if not PRONOUN.search(text):
        return "no_you_or_me"
    return None


# ---- step 2: weak labeling rules ----------------------------------------------

CONTEXT_RULES = [
    ("gaslighting", (r"gaslight|deny|denying|counter|trivializ|diminish|withhold|blocking"
                    r"|divert|crazy|reality|question your|memory|undermine")),
    ("guilt_tripping", r"guilt"),
    ("love_bombing", r"love.?bomb|lavish|soulmate|too fast|future"),
    ("breadcrumbing", r"breadcrumb"),
    ("controlling", r"control|monitor|isolat|jealous"),
]

TEXT_RULES = [
    ("guilt_tripping", (r"if you (really )?(loved|cared)|after (all|everything) i|"
                       r"i guess i'?m (just )?not|i sacrificed|you never make time|"
                       r"look what you made me")),
    ("gaslighting", (r"never happened|making (this|it|that) up|too sensitive|crazy|imagining|"
                    r"overreact|paranoid|memory|i never said|just a joke|you'?re confused|"
                    r"that'?s not what happened|overthinking")),
    ("controlling", (r"where are you|who are you with|who were you|call me (now|back)|"
                    r"answer (me|your phone)|why aren'?t you|send me a pic|"
                    r"you'?re not (going|wearing)")),
    ("love_bombing", (r"soulmate|never (met|felt) (anyone|like this)|perfect|"
                     r"the only (one|person)|meant to be|love everything about")),
    ("breadcrumbing", (r"\bwyd\b|\bu up\b|crossed my mind|thinking (about|of) you|"
                      r"we should (hang|link|get)|maybe (next|another)")),
    ("disrespect", (r"stupid|dumb|ugly|loser|shut up|nobody (would|else)|"
                   r"you'?re lucky|embarrass")),
]


def _first_match(rules, text: str) -> str | None:
    for label, pattern in rules:
        if re.search(pattern, text or "", re.IGNORECASE):
            return label
    return None


def weak_label(text: str, context: str) -> tuple[str | None, str]:
    """Return (label, method). label None = a human needs to decide."""
    text = text.replace("’", "'")
    by_context = _first_match(CONTEXT_RULES, context)
    by_text = _first_match(TEXT_RULES, text)
    if by_context and by_text:
        if by_context == by_text:
            return by_context, "rule_agree"
        return None, "rule_conflict"
    if by_context:
        return by_context, "rule_context"
    if by_text:
        return by_text, "rule_text"
    return None, "no_rule"


# ---- pipeline -----------------------------------------------------------------

def run(raw_path: Path, out_dir: Path) -> dict:
    df = pd.read_csv(raw_path)
    df["context"] = df["context"].fillna("")
    df["drop_reason"] = [drop_reason(t, c) for t, c in zip(df.text, df.context)]
    kept = df[df.drop_reason.isna()].copy()

    labels = [weak_label(t, c) for t, c in zip(kept.text, kept.context)]
    kept["label"] = [lab for lab, _ in labels]
    kept["label_method"] = [m for _, m in labels]

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["text", "label", "label_method", "context", "source_url"]
    labeled = kept[kept.label.notna()][cols]
    review = kept[kept.label.isna()][cols].assign(label="")
    labeled.to_csv(out_dir / "scraped_weak_labels.csv", index=False)
    review.to_csv(out_dir / "review_me.csv", index=False)
    df[df.drop_reason.notna()][["text", "drop_reason", "context"]].to_csv(
        out_dir / "dropped.csv", index=False)

    return {
        "raw": len(df),
        "dropped": df.drop_reason.value_counts().to_dict(),
        "kept": len(kept),
        "auto_labeled": len(labeled),
        "by_method": labeled.label_method.value_counts().to_dict(),
        "by_label": labeled.label.value_counts().to_dict(),
        "needs_review": len(review),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = run(Path("data/raw/article_quotes.csv"), Path("data/interim"))
    print(f"📥 raw quotes:      {s['raw']}")
    print(f"🗑️  dropped:         {sum(s['dropped'].values())}  {s['dropped']}")
    print(f"✅ kept:            {s['kept']}")
    print(f"🏷️  auto-labeled:    {s['auto_labeled']}  {s['by_method']}")
    print(f"     per label:     {s['by_label']}")
    print(f"👀 needs review:    {s['needs_review']}  -> data/interim/review_me.csv")


if __name__ == "__main__":
    main()
