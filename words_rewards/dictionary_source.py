"""Scraping layer for dictionary.com.

Only two pages are needed: the Word of the Day landing page (a source of
words) and ``/browse/<word>`` (the definitions for one word).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

from words_rewards.config import Settings
from words_rewards.models import Sense, WordEntry

_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([;,.:!?])")


class DictionaryError(RuntimeError):
    """Raised when dictionary.com cannot be reached or parsed."""


class WordNotFoundError(DictionaryError):
    """Raised when dictionary.com has no entry for the requested word."""


def _clean(text: str) -> str:
    """Collapse whitespace, including the gaps left by inline markup."""
    collapsed = _WHITESPACE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", collapsed)


def _slugify(word: str) -> str:
    return quote(_clean(word).lower().replace(" ", "-"))


def _examples(node) -> list[str]:
    found = [_clean(example.get_text(" ")) for example in node.select(".txt-example")]
    return [example for example in found if example]


def _senses_from_item(item, part_of_speech: str) -> list[Sense]:
    """Flatten one ``li.item-definition`` into senses.

    Some entries put a subject label ("Physics.") in the item and the real
    definitions in a nested ``ol.list-sub-definition``; those become one sense
    each, carrying the label as a prefix.
    """
    paragraph = item.find("p", recursive=False) or item.find("p")
    label = _clean(paragraph.get_text(" ")) if paragraph else ""

    sub_items = item.select("ol.list-sub-definition > li.item-sub-definition")
    if not sub_items:
        if not label:
            return []
        return [
            Sense(
                part_of_speech=part_of_speech,
                definition=label,
                examples=_examples(item),
            )
        ]

    senses: list[Sense] = []
    for sub_item in sub_items:
        sub_paragraph = sub_item.find("p")
        if sub_paragraph is None:
            continue
        definition = _clean(sub_paragraph.get_text(" "))
        if not definition:
            continue
        senses.append(
            Sense(
                part_of_speech=part_of_speech,
                definition=f"{label} {definition}".strip(),
                examples=_examples(sub_item),
            )
        )
    return senses


@dataclass(frozen=True)
class WordOfTheDay:
    """A teaser from the Word of the Day landing page."""

    word: str
    date: str
    part_of_speech: str
    short_definition: str


class DictionaryClient:
    """Fetches and parses dictionary.com pages."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    # ------------------------------------------------------------------ HTTP

    def _get(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.settings.http_retries):
            try:
                response = self.session.get(
                    url, timeout=self.settings.http_timeout, allow_redirects=True
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 404:
                    raise WordNotFoundError(f"dictionary.com has no page at {url}")
                if response.ok:
                    return response.text
                last_error = DictionaryError(
                    f"GET {url} returned HTTP {response.status_code}"
                )
            if attempt + 1 < self.settings.http_retries:
                time.sleep(2**attempt)
        raise DictionaryError(f"could not fetch {url}: {last_error}")

    # --------------------------------------------------------------- parsing

    def parse_entry(self, html: str, url: str, fallback_word: str = "") -> WordEntry:
        """Turn a ``/browse/<word>`` page into a :class:`WordEntry`."""
        soup = BeautifulSoup(html, "html.parser")

        headword = soup.select_one("h1.hdr-headword") or soup.select_one("h1")
        word = _clean(headword.get_text(" ")) if headword else fallback_word
        if not word:
            raise DictionaryError(f"no headword found on {url}")

        ipa = soup.select_one(".txt-ipa")
        pronunciation = _clean(ipa.get_text(" ")) if ipa else ""

        senses: list[Sense] = []
        for block in soup.select("div.box-posb"):
            label = block.select_one(".box-pos-label")
            part_of_speech = _clean(label.get_text(" ")) if label else ""
            for item in block.select("li.item-definition"):
                senses.extend(_senses_from_item(item, part_of_speech))
                if len(senses) >= self.settings.max_senses:
                    break
            if len(senses) >= self.settings.max_senses:
                break
        del senses[self.settings.max_senses :]

        if not senses:
            # Entry pages for rare words sometimes render only the meta
            # description; it always carries the primary definition.
            meta = soup.select_one('meta[name="description"]')
            summary = _clean(meta.get("content", "")) if meta else ""
            match = re.search(r"definition:\s*(.+?)(?:\s*See examples|$)", summary)
            if match:
                senses.append(Sense(definition=_clean(match.group(1))))

        if not senses:
            raise DictionaryError(f"no definitions found on {url}")

        return WordEntry(
            word=word,
            source_url=url,
            senses=senses,
            pronunciation=pronunciation,
        )

    def parse_word_of_the_day(self, html: str) -> list[WordOfTheDay]:
        soup = BeautifulSoup(html, "html.parser")
        entries: list[WordOfTheDay] = []
        for wrapper in soup.select(".wotd-entry-wrapper"):
            headword = wrapper.select_one(".wotd-entry-headword")
            if headword is None:
                continue
            date = wrapper.select_one(".wotd-entry-date")
            pos = wrapper.select_one(".wotd-entry-pos")
            definition = wrapper.select_one(".wotd-entry-definition")
            entries.append(
                WordOfTheDay(
                    word=_clean(headword.get_text(" ")),
                    date=_clean(date.get_text(" ")) if date else "",
                    part_of_speech=_clean(pos.get_text(" ")) if pos else "",
                    short_definition=(
                        _clean(definition.get_text(" ")) if definition else ""
                    ),
                )
            )
        if not entries:
            raise DictionaryError("no Word of the Day entries found")
        return entries

    # ---------------------------------------------------------------- public

    def fetch_entry(self, word: str) -> WordEntry:
        """Pull ``word`` and its meaning from dictionary.com."""
        if not _clean(word):
            raise ValueError("word must not be empty")
        url = urljoin(
            self.settings.dictionary_base_url + "/", f"browse/{_slugify(word)}"
        )
        return self.parse_entry(self._get(url), url, fallback_word=_clean(word))

    def fetch_word_of_the_day(self) -> list[WordOfTheDay]:
        """Words featured on the Word of the Day landing page, newest first."""
        url = urljoin(self.settings.dictionary_base_url + "/", "word-of-the-day")
        return self.parse_word_of_the_day(self._get(url))

    def random_word(self, rng: random.Random | None = None) -> str:
        """Pick one of the featured Word of the Day words at random."""
        candidates = self.fetch_word_of_the_day()
        chooser = rng or random
        return chooser.choice(candidates).word
