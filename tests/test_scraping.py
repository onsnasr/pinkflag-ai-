"""Scraper tests run on a saved fake page: no internet needed."""
import requests

from pinkflag.scraping.fetcher import PoliteFetcher
from pinkflag.scraping.oembed import split_hashtags
from pinkflag.scraping.parser import parse_article

ARTICLE = """
<html><body>
<nav><p>"Subscribe to our newsletter today"</p></nav>
<article>
  <h2>Gaslighting</h2>
  <p>A classic one is “you're being way too sensitive” after he crosses a line.</p>
  <h2>Breadcrumbing</h2>
  <li>The 2am "wyd" text with no plans attached.</li>
  <li>"u up?"</li>
  <blockquote class="tiktok-embed"
     cite="https://www.tiktok.com/@someone/video/7234567890123456789">caption here</blockquote>
  <p>Read more at <a href="https://www.tiktok.com/@other.user/video/111?lang=en">this</a>.</p>
  <p>He said "ok" once.</p>
</article>
</body></html>
"""


def test_extracts_quotes_with_heading_context():
    examples, _ = parse_article(ARTICLE)
    texts = {e.text: e.context for e in examples}
    assert texts["you're being way too sensitive"] == "Gaslighting"
    assert texts["u up?"] == "Breadcrumbing"


def test_skips_navigation_and_one_word_quotes():
    examples, _ = parse_article(ARTICLE)
    texts = [e.text for e in examples]
    assert not any("newsletter" in t for t in texts)
    assert "ok" not in texts


def test_finds_tiktok_links_from_embeds_and_anchors():
    _, tiktoks = parse_article(ARTICLE)
    assert tiktoks == [
        "https://www.tiktok.com/@someone/video/7234567890123456789",
        "https://www.tiktok.com/@other.user/video/111",
    ]


def test_split_hashtags():
    text, tags = split_hashtags("if he says this RUN 🚩 #RedFlags #datingadvice")
    assert text == "if he says this RUN 🚩"
    assert tags == ["redflags", "datingadvice"]


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code, self.text = status_code, text
        self.ok = 200 <= status_code < 300

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(self.status_code)


class FakeSession(requests.Session):
    def __init__(self, robots_txt):
        super().__init__()
        self.robots_txt, self.calls = robots_txt, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            return FakeResponse(200, self.robots_txt)
        return FakeResponse(200, "<html>page</html>")


def test_robots_txt_is_respected(tmp_path):
    session = FakeSession("User-agent: *\nDisallow: /private/\n")
    f = PoliteFetcher("test-bot", delay_s=0, cache_dir=tmp_path, session=session)
    assert f.get("https://example.com/private/secret") is None
    assert f.get("https://example.com/public/post") == "<html>page</html>"
    assert f.stats["blocked_by_robots"] == 1


def test_cache_prevents_second_download(tmp_path):
    session = FakeSession("")
    f = PoliteFetcher("test-bot", delay_s=0, cache_dir=tmp_path, session=session)
    f.get("https://example.com/a")
    f.get("https://example.com/a")
    page_calls = [c for c in session.calls if not c.endswith("robots.txt")]
    assert len(page_calls) == 1
    assert f.stats["cached"] == 1
