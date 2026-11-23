"""The tools Sisi can call. Gemini decides *when*; this code decides *what happens*.

Each tool = a Python function + a declaration (name, description, parameters) that
tells Gemini what the tool does. This is "function calling".
"""
from pathlib import Path

import joblib

from pinkflag.agent.memory import Memory
from pinkflag.models.baseline import predict
from pinkflag.taxonomy import LABELS

# best first: trained on trusted labels only (66% on the clean test set, see README)
MODEL_CANDIDATES = ["models/baseline_strong.joblib", "models/baseline_merged.joblib",
                    "models/baseline.joblib"]

DECLARATIONS = [
    {
        "name": "analyze_messages",
        "description": "Run PinkFlag's red-flag detector on messages HE sent. Returns, for each "
                       "message, the most likely label (green_flag or a red-flag type) with "
                       "confidence.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"messages": {"type": "ARRAY", "items": {"type": "STRING"},
                                        "description": "His messages, one per item, verbatim."}},
            "required": ["messages"],
        },
    },
    {
        "name": "explain_flag",
        "description": "Get the definition of a PinkFlag label (e.g. gaslighting, breadcrumbing).",
        "parameters": {
            "type": "OBJECT",
            "properties": {"label": {"type": "STRING", "enum": list(LABELS)}},
            "required": ["label"],
        },
    },
    {
        "name": "remember",
        "description": "Save a short, durable fact about her situation for future conversations "
                       "(his name, how long they've talked, a notable event, her feelings or "
                       "boundaries). One fact per call.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"note": {"type": "STRING"}},
            "required": ["note"],
        },
    },
    {
        "name": "safety_resources",
        "description": "Use when she may be unsafe: threats, violence, stalking, sexual pressure, "
                       "or self-harm. Returns safety guidance to share.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]


class Toolbox:
    def __init__(self, memory: Memory, model_path: str | None = None):
        self.memory = memory
        path = model_path or next((p for p in MODEL_CANDIDATES if Path(p).exists()), None)
        self.model = joblib.load(path) if path else None
        self.model_name = Path(path).stem if path else None

    def analyze_messages(self, messages: list[str]) -> dict:
        if self.model is None:
            return {"error": "no trained model found; run python -m pinkflag.train first"}
        results = []
        for v in predict(self.model, [m for m in messages if m.strip()]):
            label, conf = v.top_labels[0]
            results.append({"message": v.text, "label": label, "display": LABELS[label].display,
                            "confidence": round(conf, 2),
                            "runner_up": v.top_labels[1][0] if len(v.top_labels) > 1 else None})
        return {"model": self.model_name, "results": results}

    def explain_flag(self, label: str) -> dict:
        lab = LABELS.get(label)
        if lab is None:
            return {"error": f"unknown label {label}"}
        return {"label": label, "name": lab.display, "emoji": lab.emoji,
                "definition": lab.definition, "is_red_flag": lab.is_red_flag}

    def remember(self, note: str) -> dict:
        return {"saved": self.memory.remember(note)}

    def safety_resources(self) -> dict:
        return {"guidance": [
            "If you are in immediate danger, call your local emergency number now.",
            "Tell someone you trust where you are and what is happening.",
            "Save evidence (screenshots, dates) somewhere he can't access.",
            "Local domestic-violence or women's support lines can help confidentially.",
            "It is not your fault.",
        ]}

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if fn is None or name not in {d["name"] for d in DECLARATIONS}:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as err:
            return {"error": f"bad arguments: {err}"}
