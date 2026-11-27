"""Deploy PinkFlag to a Hugging Face Space (free public link).

    python tools/deploy_space.py

Needs HF_TOKEN in .env (a *write* token from huggingface.co/settings/tokens).
Uploads the code + the trained detector + Sisi's voice to <your-username>/pinkflag.
The Gemini key is NOT uploaded: add it yourself in the Space settings as a secret
named GEMINI_API_KEY.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import HfApi

from pinkflag.env import get_key

SPACE_README = """---
title: PinkFlag
emoji: 🎀
colorFrom: pink
colorTo: red
sdk: docker
app_port: 7860
pinned: true
short_description: Your AI big sister for safer dating
---

# 🎀 PinkFlag: your AI big sister for safer dating

Chat with **Sisi**: paste what he said and get the receipts. A classifier we trained detects
manipulation patterns; Sisi explains them. Code, data pipeline and model comparison:
https://github.com/onsnasr/pinkflag-ai-

Not a therapist, not a verdict on anyone. If you feel unsafe, contact local emergency services.
"""

FILES = ["Dockerfile", "pyproject.toml", "src", "web", "configs/agent.yaml",
         "models/baseline_strong.joblib", "models/voices"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    api = HfApi(token=get_key("HF_TOKEN", ROOT / ".env"))
    user = api.whoami()["name"]
    repo = f"{user}/pinkflag"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in FILES:
            src, dst = ROOT / item, stage / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
            else:
                shutil.copy2(src, dst)
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.upload_folder(folder_path=stage, repo_id=repo, repo_type="space",
                          commit_message="Deploy PinkFlag")
    print(f"✅ uploaded. Space: https://huggingface.co/spaces/{repo}")
    print("👉 last step: Space → Settings → Variables and secrets → New secret: GEMINI_API_KEY")


if __name__ == "__main__":
    main()
