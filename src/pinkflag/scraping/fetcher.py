"""A polite HTTP fetcher.

Three rules, enforced in code so nobody can forget them:
  1. robots.txt   -> if a site says bots may not read a page, we don't.
  2. delay        -> wait between requests to the same site, never hammer it.
  3. cache        -> every page is saved to disk, so it's downloaded only once.
"""
import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


class PoliteFetcher:
    def __init__(self, user_agent: str, delay_s: float = 2.0,
                 cache_dir: str | Path = "data/raw/cache", timeout_s: float = 20.0,
                 session: requests.Session | None = None):
        self.user_agent = user_agent
        self.delay_s = delay_s
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._robots: dict[str, RobotFileParser] = {}
        self._last_hit: dict[str, float] = {}
        self.stats = {"fetched": 0, "cached": 0, "blocked_by_robots": 0, "failed": 0}

    # ---- rule 1: robots.txt -------------------------------------------------
    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._robots:
            rp = RobotFileParser()
            try:
                resp = self.session.get(f"{root}/robots.txt", timeout=self.timeout_s)
                # no robots.txt (404) means no restrictions; errors (401/403/5xx) mean "stay out"
                if resp.status_code == 404:
                    rp.parse([])
                elif resp.ok:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.disallow_all = True
            except requests.RequestException:
                rp.disallow_all = True
            self._robots[root] = rp
        return self._robots[root].can_fetch(self.user_agent, url)

    # ---- rule 2: delay ------------------------------------------------------
    def _wait_turn(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last_hit.get(host, 0.0)
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_hit[host] = time.monotonic()

    # ---- rule 3: cache ------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".txt")

    def get(self, url: str) -> str | None:
        """Return the page text, or None if not allowed / failed."""
        cached = self._cache_path(url)
        if cached.exists():
            self.stats["cached"] += 1
            return cached.read_text(encoding="utf-8")

        if not self.allowed(url):
            self.stats["blocked_by_robots"] += 1
            return None

        self._wait_turn(url)
        try:
            resp = self.session.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
        except requests.RequestException:
            self.stats["failed"] += 1
            return None

        self.stats["fetched"] += 1
        cached.write_text(resp.text, encoding="utf-8")
        return resp.text
