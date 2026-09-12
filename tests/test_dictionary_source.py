from __future__ import annotations

import pytest

from words_rewards.dictionary_source import (
    DictionaryClient,
    DictionaryError,
    WordNotFoundError,
    _clean,
    _slugify,
)


def test_clean_collapses_whitespace_and_markup_gaps():
    assert _clean("  good fortune ;  luck .  ") == "good fortune; luck."


@pytest.mark.parametrize(
    ("word", "expected"),
    [("Serendipity", "serendipity"), ("de facto", "de-facto")],
)
def test_slugify(word, expected):
    assert _slugify(word) == expected


def test_fetch_entry_parses_senses(offline_client):
    entry = offline_client.fetch_entry("serendipity")

    assert entry.word == "serendipity"
    assert entry.source_url == "https://www.dictionary.com/browse/serendipity"
    assert entry.pronunciation
    assert entry.parts_of_speech == ["noun"]
    assert entry.senses[0].definition == (
        "an aptitude for making desirable discoveries by accident."
    )
    assert any(sense.examples for sense in entry.senses)


def test_meaning_is_a_numbered_block(entry):
    meaning = entry.meaning
    assert meaning.startswith("1. (noun) an aptitude for making")
    assert "example: " in meaning


def test_max_senses_is_respected(settings, offline_client, entry_html):
    client = DictionaryClient(
        settings.__class__(**{**settings.__dict__, "max_senses": 2}),
        session=offline_client.session,
    )
    assert len(client.fetch_entry("serendipity").senses) == 2


def test_unknown_word_raises(offline_client):
    with pytest.raises(WordNotFoundError):
        offline_client.fetch_entry("notarealword")


def test_empty_word_rejected(offline_client):
    with pytest.raises(ValueError):
        offline_client.fetch_entry("   ")


def test_word_of_the_day_is_parsed(offline_client):
    featured = offline_client.fetch_word_of_the_day()

    assert featured
    assert featured[0].word
    assert all(item.word for item in featured)


def test_random_word_comes_from_the_featured_list(offline_client):
    import random

    featured = {item.word for item in offline_client.fetch_word_of_the_day()}
    assert offline_client.random_word(random.Random(0)) in featured


def test_parse_entry_without_definitions_raises(offline_client):
    with pytest.raises(DictionaryError):
        offline_client.parse_entry("<html><h1>ghost</h1></html>", "http://x")


def test_parse_entry_falls_back_to_meta_description(offline_client):
    html = (
        '<html><head><meta name="description" content="GHOST definition: a '
        'disembodied soul. See examples of ghost used in a sentence."></head>'
        "<body><h1 class='hdr-headword'>ghost</h1></body></html>"
    )
    entry = offline_client.parse_entry(html, "http://x")
    assert entry.senses[0].definition == "a disembodied soul."
