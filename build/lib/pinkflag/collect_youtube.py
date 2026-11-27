"""Collect public YouTube comments on Tunisian relationship videos (official API).

    python -m pinkflag.collect_youtube --config configs/youtube.yaml

Uses the YouTube Data API v3 with your own key (YOUTUBE_API_KEY in .env).
Privacy: we keep the comment text only, never the commenter's name or channel,
and mask @mentions / phone numbers / links. YouTube's API policies require
stored data to be refreshed or deleted within 30 days, so data/raw/ is temporary.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

from pinkflag.datasets.tunisian import clean, script_of
from pinkflag.env import get_key

API = "https://www.googleapis.com/youtube/v3"


def search_videos(session, key: str, query: str, region: str, n: int) -> list[dict]:
    resp = session.get(f"{API}/search", params={
        "key": key, "part": "snippet", "q": query, "type": "video",
        "regionCode": region, "maxResults": min(n, 50)}, timeout=30)
    resp.raise_for_status()
    return [{"video_id": it["id"]["videoId"], "title": it["snippet"]["title"]}
            for it in resp.json().get("items", [])]


def parse_comments(payload: dict) -> list[str]:
    out = []
    for item in payload.get("items", []):
        top = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
        out.append(top)
        for reply in item.get("replies", {}).get("comments", []):
            out.append(reply["snippet"]["textOriginal"])
    return out


def video_comments(session, key: str, video_id: str, limit: int) -> list[str]:
    comments, page = [], None
    while len(comments) < limit:
        params = {"key": key, "part": "snippet,replies", "videoId": video_id,
                  "maxResults": 100, "textFormat": "plainText"}
        if page:
            params["pageToken"] = page
        resp = session.get(f"{API}/commentThreads", params=params, timeout=30)
        if resp.status_code == 403:  # comments disabled on this video
            break
        resp.raise_for_status()
        data = resp.json()
        comments += parse_comments(data)
        page = data.get("nextPageToken")
        if not page:
            break
    return comments[:limit]


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Collect Tunisian YouTube comments")
    parser.add_argument("--config", default="configs/youtube.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    key = get_key("YOUTUBE_API_KEY")
    session = requests.Session()

    videos = {v: "(hand-picked)" for v in cfg.get("extra_video_ids") or []}
    for q in cfg["queries"]:
        for v in search_videos(session, key, q, cfg["region"], cfg["videos_per_query"]):
            videos.setdefault(v["video_id"], v["title"])
        print(f"  🔎 '{q}' -> {len(videos)} videos so far")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for vid, title in videos.items():
        texts = video_comments(session, key, vid, cfg["comments_per_video"])
        rows += [{"text": clean(t), "video_id": vid, "video_title": title,
                  "collected_at": now} for t in texts]
        print(f"  💬 {len(texts):4d} comments  {title[:60]}")

    df = pd.DataFrame(rows)
    df = df[df.text.str.len() >= 3].drop_duplicates("text")
    df["script"] = df.text.map(script_of)
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ {len(df)} unique comments from {len(videos)} videos -> {out}")
    print(f"   by script: {df.script.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
