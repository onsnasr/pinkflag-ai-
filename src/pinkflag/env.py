"""Read API keys from environment variables or a local .env file (never committed)."""
import os
from pathlib import Path


def get_key(name: str, env_file: str | Path = ".env") -> str:
    if os.environ.get(name):
        return os.environ[name]
    path = Path(env_file)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit(f"❌ Missing {name}. Put it in .env (see .env.example).")
