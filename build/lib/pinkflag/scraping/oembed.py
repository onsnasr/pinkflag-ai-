"""TikTok captions through TikTok's official oEmbed endpoint.

oEmbed is the public API TikTok provides so websites can embed its videos.
We give it a video URL, it returns JSON with the caption ("title") and creator.
No scraping of tiktok.com pages, no logins, no bot-protection tricks.
"""
import json
import re
from dataclasses import dataclass
from urllib.parse import quote

from pinkflag.scraping.fetcher import PoliteFetcher

OEMBED_URL = "https://www.tiktok.com/oembed?url={}"
HASHTAG_RE = re.compile(r"#(\w+)")


@dataclass
class TikTokCaption:
    video_url: str
    caption: str
    hashtags: list[str]
    author: str


def split_hashtags(caption: str) -> tuple[str, list[str]]:
    tags = [t.lower() for t in HASHTAG_RE.findall(caption)]
    text = re.sub(r"\s+", " ", HASHTAG_RE.sub("", caption)).strip()
    return text, tags


def fetch_caption(fetcher: PoliteFetcher, video_url: str) -> TikTokCaption | None:
    raw = fetcher.get(OEMBED_URL.format(quote(video_url, safe="")))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if "title" not in data:
        return None
    text, tags = split_hashtags(data.get("title", ""))
    return TikTokCaption(video_url=video_url, caption=text, hashtags=tags,
                         author=data.get("author_name", ""))
