"""Beautiful Soup parsing: turn an article's HTML into example messages + TikTok links.

Articles about red-flag texts usually look like:

    <h2>Gaslighting</h2>
    <p>If he says “you're being too sensitive” after...</p>
    <blockquote class="tiktok-embed" cite="https://www.tiktok.com/@user/video/123">

So we pull out:
  - quoted phrases ("you're being too sensitive")  -> candidate messages
  - the heading above each quote ("Gaslighting")   -> a hint for labeling later
  - embedded TikTok video links                    -> sent to oEmbed for captions
"""
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# text between straight or curly double quotes
QUOTE_RE = re.compile(r'[“"]([^“”"]{3,220})[”"]')
TIKTOK_VIDEO_RE = re.compile(r"https?://(?:www\.)?tiktok\.com/@[\w.\-]+/video/\d+")

NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form",
              "iframe", "svg"]
TEXT_TAGS = ["p", "li", "h2", "h3", "h4", "blockquote", "td"]
HEADING_TAGS = {"h2", "h3", "h4"}


@dataclass
class Example:
    text: str
    context: str  # nearest heading above the quote: a labeling hint, not a label


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_message(text: str) -> bool:
    words = text.split()
    if not 2 <= len(words) <= 35:
        return False
    return not text.lower().startswith(("http", "www."))


def extract_tiktok_links(soup: BeautifulSoup) -> list[str]:
    links = []
    for bq in soup.select("blockquote.tiktok-embed"):
        if bq.get("cite"):
            links.append(bq["cite"])
    for a in soup.find_all("a", href=True):
        links.append(a["href"])
    found = []
    for link in links:
        m = TIKTOK_VIDEO_RE.search(link)
        if m and m.group(0) not in found:
            found.append(m.group(0))
    return found


def parse_article(html: str) -> tuple[list[Example], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    tiktoks = extract_tiktok_links(soup)  # before noise removal: embeds can sit in asides

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for bq in soup.select("blockquote.tiktok-embed"):
        bq.decompose()  # its text is the caption, which oEmbed gives us cleanly

    body = soup.find("article") or soup.find("main") or soup.body or soup
    examples, seen = [], set()
    heading = ""
    for el in body.find_all(TEXT_TAGS):
        text = _clean(el.get_text(" "))
        if el.name in HEADING_TAGS:
            heading = text[:120]
        for quote in QUOTE_RE.findall(text):
            quote = _clean(quote).strip(" .,")
            key = quote.lower()
            if _looks_like_message(quote) and key not in seen:
                seen.add(key)
                examples.append(Example(text=quote, context=heading))
    return examples, tiktoks
