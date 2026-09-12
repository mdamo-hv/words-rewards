from __future__ import annotations

from pathlib import Path

import pytest

from words_rewards.config import Settings
from words_rewards.dictionary_source import DictionaryClient
from words_rewards.models import WordEntry

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class FakeSession:
    """Serves canned HTML instead of hitting dictionary.com."""

    def __init__(self, routes: dict[str, FakeResponse]) -> None:
        self.headers: dict[str, str] = {}
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url: str, **_kwargs) -> FakeResponse:
        self.requested.append(url)
        return self.routes.get(url, FakeResponse(status_code=404))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(output_dir=tmp_path / "results", http_retries=1)


@pytest.fixture
def entry_html() -> str:
    return (FIXTURES / "browse_serendipity.html").read_text(encoding="utf-8")


@pytest.fixture
def wotd_html() -> str:
    return (FIXTURES / "word_of_the_day.html").read_text(encoding="utf-8")


@pytest.fixture
def offline_client(settings, entry_html, wotd_html) -> DictionaryClient:
    routes = {
        "https://www.dictionary.com/browse/serendipity": FakeResponse(entry_html),
        "https://www.dictionary.com/word-of-the-day": FakeResponse(wotd_html),
    }
    return DictionaryClient(settings, session=FakeSession(routes))


@pytest.fixture
def entry(offline_client) -> WordEntry:
    return offline_client.fetch_entry("serendipity")
