"""Publish the trained detector to the Hugging Face Hub (free public model repo).

    python tools/publish_model.py

The hosted app downloads it at startup (PINKFLAG_MODEL_REPO), so the model never goes in git.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sklearn
from huggingface_hub import HfApi

from pinkflag.env import get_key

CARD = """---
license: mit
language: [en]
tags: [text-classification, scikit-learn, manipulation-detection]
---
# PinkFlag detector (TF-IDF + Logistic Regression)

Classifies a dating message into 8 labels (green_flag, love_bombing, gaslighting, controlling,
guilt_tripping, disrespect, intimidation, breadcrumbing). Trained on 591 trusted-label examples.
Clean test set: 61% accuracy (52-69, 95% CI), red flag vs green 97%.
Full model card, data and comparison with a BiLSTM and XLM-RoBERTa:
https://github.com/onsnasr/pinkflag-ai-

Load: `joblib.load("baseline_strong.joblib")` (scikit-learn {sk}). Not a judgment of any person.
"""


def main() -> None:
    api = HfApi(token=get_key("HF_TOKEN", ROOT / ".env"))
    repo = f"{api.whoami()['name']}/pinkflag-detector"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(ROOT / "models/baseline_strong.joblib"),
                    path_in_repo="baseline_strong.joblib", repo_id=repo)
    api.upload_file(path_or_fileobj=CARD.replace("{sk}", sklearn.__version__).encode(),
                    path_in_repo="README.md", repo_id=repo)
    print(f"https://huggingface.co/{repo}  (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
