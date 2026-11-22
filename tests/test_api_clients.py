"""API clients tested on fake responses: no keys or internet needed."""
import pandas as pd

from pinkflag.collect_youtube import parse_comments
from pinkflag.env import get_key
from pinkflag.translate_derja import build_prompt, parse_response


def test_parse_youtube_comments_with_replies():
    payload = {"items": [
        {"snippet": {"topLevelComment": {"snippet": {"textOriginal": "rajel toxique 100%"}}},
         "replies": {"comments": [{"snippet": {"textOriginal": "sa7a 3lik"}}]}},
        {"snippet": {"topLevelComment": {"snippet": {"textOriginal": "brabi chkoun hedha"}}}},
    ]}
    assert parse_comments(payload) == ["rajel toxique 100%", "sa7a 3lik", "brabi chkoun hedha"]


def test_gemini_prompt_and_parse():
    batch = pd.DataFrame({"text": ["where are you rn"], "label": ["controlling"]}, index=[7])
    prompt = build_prompt(batch, variants=2)
    assert "7: where are you rn" in prompt and "2 different" in prompt
    payload = {"candidates": [{"content": {"parts": [
        {"text": '```json\n[{"id": 7, "derja": ["win enty tw", "winek taw"]}]\n```'}]}}]}
    assert parse_response(payload) == [{"id": 7, "derja": ["win enty tw", "winek taw"]}]


def test_get_key_reads_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('FAKE_KEY="abc123"\n', encoding="utf-8")
    assert get_key("FAKE_KEY", env) == "abc123"
