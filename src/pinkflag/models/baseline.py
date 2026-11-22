"""Baseline model: TF-IDF features + Logistic Regression.

Why start here? It trains in seconds, it's explainable (we can see which words
pushed a prediction), and every fancier model has to beat it to earn its place.

One model predicts the type (green_flag, gaslighting, ...). The red-flag score is
derived from it: P(red flag) = 1 - P(green_flag). Using one model means the
"is it a red flag" answer and the "what kind" answer can never contradict each other.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from pinkflag.taxonomy import GREEN, LABELS


def build_pipeline(cfg: dict) -> Pipeline:
    f, m = cfg["features"], cfg["model"]
    features = FeatureUnion([
        ("words", TfidfVectorizer(
            ngram_range=tuple(f["word_ngrams"]), sublinear_tf=f["sublinear_tf"])),
        ("chars", TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(f["char_ngrams"]),
            sublinear_tf=f["sublinear_tf"])),
    ])
    clf = LogisticRegression(
        C=m["C"], class_weight=m["class_weight"], max_iter=m["max_iter"],
        random_state=cfg.get("seed"))
    return Pipeline([("tfidf", features), ("clf", clf)])


@dataclass
class Verdict:
    text: str
    red_flag_prob: float
    top_labels: list[tuple[str, float]]

    @property
    def is_red_flag(self) -> bool:
        # Not "red_flag_prob >= 0.5": with 6 red types vs 1 green, the red share sums
        # small slices from every type and almost always passes 50%. Instead we trust
        # the single most likely label, so the verdict always matches the top type.
        return self.top_labels[0][0] != GREEN

    def pretty(self) -> str:
        head = "🚩 RED FLAG" if self.is_red_flag else "💚 seems fine"
        lines = [f'"{self.text}"',
                 f"{head}  (all red-flag types combined: {self.red_flag_prob:.0%})"]
        for key, p in self.top_labels:
            lab = LABELS[key]
            lines.append(f"   {lab.emoji} {lab.display:28s} {p:5.0%} {'█' * int(p * 25)}")
        return "\n".join(lines)


def predict(model: Pipeline, texts: list[str], top_k: int = 3) -> list[Verdict]:
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    green_idx = classes.index(GREEN)
    out = []
    for text, p in zip(texts, probs):
        order = np.argsort(p)[::-1][:top_k]
        out.append(Verdict(
            text=text,
            red_flag_prob=float(1 - p[green_idx]),
            top_labels=[(classes[i], float(p[i])) for i in order],
        ))
    return out


def top_features(model: Pipeline, n: int = 8) -> dict[str, list[str]]:
    """Strongest word features per label: what the model actually learned."""
    words = model.named_steps["tfidf"].transformer_list[0][1]
    vocab = words.get_feature_names_out()
    clf = model.named_steps["clf"]
    result = {}
    for label, coefs in zip(clf.classes_, clf.coef_):
        word_coefs = coefs[: len(vocab)]  # skip char n-grams, they're unreadable
        result[label] = [vocab[i] for i in np.argsort(word_coefs)[::-1][:n]]
    return result
