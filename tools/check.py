"""Run the whole quality + security pipeline locally, before pushing.

    python tools/check.py          # lint, unit, e2e, SAST, SCA
    python tools/check.py --fast   # lint + unit only

DAST (OWASP ZAP) needs Docker, so it runs in GitHub Actions only.
"""
import os
import subprocess
import sys

FAST = "--fast" in sys.argv
STEPS = [
    ("lint (ruff)", ["ruff", "check", "."]),
    ("unit + API tests", [sys.executable, "-m", "pytest", "-q", "--ignore=tests/e2e"]),
    ("end-to-end (Playwright)", [sys.executable, "-m", "pytest", "-q", "tests/e2e"]),
    ("SAST (Bandit)", ["bandit", "-q", "-r", "src", "-ll"]),
    ("SCA (pip-audit)", ["pip-audit", "--skip-editable", "--progress-spinner", "off"]),
]
if FAST:
    STEPS = STEPS[:2]

env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
results = []
for name, cmd in STEPS:
    print(f"\n=== {name} ===", flush=True)
    try:
        ok = subprocess.run(cmd, check=False, env=env).returncode == 0
    except FileNotFoundError:
        ok = False
        print(f"{cmd[0]} is not installed: pip install {cmd[0]}")
    results.append((name, ok))

print("\n=== summary ===")
for name, ok in results:
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(ok for _, ok in results) else 1)
