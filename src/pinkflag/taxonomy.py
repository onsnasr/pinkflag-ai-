"""The label set: what PinkFlag can detect.

Each label has a Gen Z display name (for the app) and a plain definition
(for the people labeling data). If two labelers would disagree on a message,
the definition is too vague: fix the definition, not the label.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    key: str
    emoji: str
    display: str
    definition: str
    is_red_flag: bool = True


LABELS: dict[str, Label] = {
    label.key: label
    for label in [
        Label(
            "green_flag", "💚", "green flag fr",
            "Respectful, consistent, accountable communication. Respects plans, "
            "boundaries and your time.",
            is_red_flag=False,
        ),
        Label(
            "love_bombing", "💣", "love bombing",
            "Excessive, fast-moving affection or commitment (soulmate talk, future "
            "plans, gifts) far ahead of how well you actually know each other.",
        ),
        Label(
            "gaslighting", "🔥", "gaslighting",
            "Denying or rewriting events, or questioning your memory or sanity, so "
            "you doubt your own perception.",
        ),
        Label(
            "controlling", "⛓️", "controlling",
            "Monitoring, restricting or demanding: location, phone, clothes, friends, "
            "reply speed, permission for plans.",
        ),
        Label(
            "guilt_tripping", "😢", "guilt tripping",
            "Using guilt, self-pity, score-keeping or threats about their own feelings "
            "to pressure you into doing what they want.",
        ),
        Label(
            "disrespect", "🙄", "disrespectful",
            "Insults, negging, mocking, comparing you to others, or dismissing your "
            "feelings with contempt.",
        ),
        Label(
            "intimidation", "😡", "intimidation / anger",
            "Threats, angry outbursts or scare tactics meant to make you comply: "
            "yelling in caps, 'you'll regret this', threats to leave, expose or hurt.",
        ),
        Label(
            "breadcrumbing", "🍞", "breadcrumbing / time waster",
            "Low-effort, inconsistent attention that keeps you interested without "
            "commitment: late-night 'wyd', vague plans, disappearing and reappearing.",
        ),
    ]
}

GREEN = "green_flag"
RED_FLAG_LABELS = [k for k, v in LABELS.items() if v.is_red_flag]


def validate(labels) -> None:
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in data: {unknown}. Known: {sorted(LABELS)}")
