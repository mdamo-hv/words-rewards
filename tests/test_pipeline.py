from __future__ import annotations

import json

import pytest

from words_rewards.dictionary_source import WordNotFoundError
from words_rewards.mock import OverlapJudge, TemplateExplainer
from words_rewards.models import Explanation, JudgeVerdict
from words_rewards.pipeline import Pipeline
from words_rewards.sourcing import DirectWordSource
from words_rewards.storage import INDEX_FILENAME, load_index


class StubSource:
    deterministic = False

    def __init__(self, entries):
        self.entries = list(entries)
        self.calls = 0

    def pull(self):
        self.calls += 1
        return self.entries[min(self.calls - 1, len(self.entries) - 1)]


class ConstantJudge:
    model = "constant-judge"

    def __init__(self, score):
        self.score = score

    def judge(self, entry, explanation) -> JudgeVerdict:
        return JudgeVerdict(
            semantic_accuracy=self.score,
            sense_coverage=self.score,
            precision=1.0,
            correct_primary_sense=True,
            hallucinations=[],
            missing_points=[],
            score=self.score,
            rationale="constant",
        )


@pytest.fixture
def pipeline(entry, settings):
    return Pipeline(
        source=StubSource([entry]),
        explainer=TemplateExplainer(),
        judge=OverlapJudge(),
        settings=settings,
    )


def test_run_once_produces_a_scored_record(pipeline, entry):
    record = pipeline.run_once()

    assert record.word == entry.word
    assert 0.0 <= record.score <= 1.0
    assert record.passed == (record.score >= record.pass_threshold)
    assert record.explanation.saw_dictionary_meaning is False


def test_score_is_clamped_to_the_unit_interval(entry, settings):
    pipeline = Pipeline(
        source=StubSource([entry]),
        explainer=TemplateExplainer(),
        judge=ConstantJudge(1.0),
        settings=settings,
    )
    assert pipeline.run_once().score == 1.0


def test_run_writes_a_json_file_with_word_and_score(pipeline, settings):
    records = pipeline.run(count=1)
    files = sorted(settings.output_dir.glob("*.json"))

    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["word"] == records[0].word
    assert payload["score"] == records[0].score
    assert payload["verdict"]["rationale"]
    assert payload["entry"]["source_url"].startswith("https://www.dictionary.com/")


def test_run_appends_to_the_index(pipeline, settings):
    pipeline.run(count=1)
    rows = load_index(settings.output_dir)

    assert (settings.output_dir / INDEX_FILENAME).exists()
    assert rows[0]["word"] == "serendipity"
    assert 0.0 <= rows[0]["score"] <= 1.0


def test_duplicate_words_do_not_count_twice(entry, settings):
    pipeline = Pipeline(
        source=StubSource([entry]),
        explainer=TemplateExplainer(),
        judge=OverlapJudge(),
        settings=settings,
    )
    records = pipeline.run(count=2)

    assert [record.word for record in records] == ["serendipity"]


def test_deterministic_source_does_not_retry(offline_client, settings):
    source = DirectWordSource(offline_client, word="notarealword")
    pipeline = Pipeline(
        source=source,
        explainer=TemplateExplainer(),
        judge=OverlapJudge(),
        settings=settings,
    )
    with pytest.raises(WordNotFoundError):
        pipeline.run(count=1)
    assert offline_client.session.requested.count(
        "https://www.dictionary.com/browse/notarealword"
    ) == 1


def test_explainer_can_be_shown_the_meaning(entry, settings):
    pipeline = Pipeline(
        source=StubSource([entry]),
        explainer=TemplateExplainer(),
        judge=OverlapJudge(),
        settings=settings,
        show_meaning_to_explainer=True,
    )
    assert pipeline.run_once().explanation.saw_dictionary_meaning is True


def test_run_rejects_non_positive_count(pipeline):
    with pytest.raises(ValueError):
        pipeline.run(count=0)


def test_judge_model_recorded_from_the_judge(entry, settings):
    pipeline = Pipeline(
        source=StubSource([entry]),
        explainer=TemplateExplainer(),
        judge=ConstantJudge(0.5),
        settings=settings,
    )
    assert pipeline.run_once().judge_model == "constant-judge"


def test_overlap_judge_flags_an_unrelated_explanation(entry):
    verdict = OverlapJudge().judge(
        entry,
        Explanation(
            word=entry.word,
            text="A kind of Italian pasta served with tomato sauce.",
            model="stub",
        ),
    )
    assert verdict.score < 0.3
    assert verdict.correct_primary_sense is False
