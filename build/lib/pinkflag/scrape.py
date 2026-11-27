"""Run the scraping pipeline.

    python -m pinkflag.scrape --config configs/sources.yaml

articles (Beautiful Soup) -> quoted example messages + embedded TikTok links
TikTok links (oEmbed)     -> captions + hashtags
Everything lands in data/raw/ (gitignored): raw data still needs labeling.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from pinkflag.scraping.fetcher import PoliteFetcher
from pinkflag.scraping.oembed import fetch_caption
from pinkflag.scraping.parser import parse_article


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Scrape red-flag examples")
    parser.add_argument("--config", default="configs/sources.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    fetcher = PoliteFetcher(user_agent=cfg["user_agent"], delay_s=cfg["delay_seconds"],
                            cache_dir=cfg["cache_dir"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, tiktok_links = [], []
    for url in cfg["articles"]:
        html = fetcher.get(url)
        if html is None:
            print(f"  ⏭️  skipped (robots.txt or error): {url}")
            continue
        examples, tiktoks = parse_article(html)
        print(f"  📰 {len(examples):3d} quotes, {len(tiktoks):2d} tiktoks  {url}")
        rows += [{"text": e.text, "context": e.context, "source_url": url,
                  "kind": "article_quote", "scraped_at": now} for e in examples]
        tiktok_links += [t for t in tiktoks if t not in tiktok_links]

    tiktok_links += [t for t in cfg.get("tiktok_videos") or [] if t not in tiktok_links]
    captions = []
    for link in tiktok_links:
        cap = fetch_caption(fetcher, link)
        if cap and cap.caption:
            captions.append({"text": cap.caption, "hashtags": " ".join(cap.hashtags),
                             "author": cap.author, "source_url": cap.video_url,
                             "kind": "tiktok_caption", "scraped_at": now})
    print(f"  🎵 {len(captions)} TikTok captions from {len(tiktok_links)} links")

    quotes = pd.DataFrame(rows).drop_duplicates("text") if rows else pd.DataFrame()
    quotes.to_csv(out_dir / "article_quotes.csv", index=False)
    pd.DataFrame(captions).to_csv(out_dir / "tiktok_captions.csv", index=False)

    print(f"\n✅ {len(quotes)} unique quotes -> {out_dir / 'article_quotes.csv'}")
    print(f"✅ {len(captions)} captions     -> {out_dir / 'tiktok_captions.csv'}")
    print(f"📊 fetcher: {fetcher.stats}")


if __name__ == "__main__":
    main()
