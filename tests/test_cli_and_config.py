from __future__ import annotations

import json

import pytest

from words_rewards.cli import build_parser, main, settings_from_args, summarise
from words_rewards.config import Settings
from words_rewards.models import JudgeVerdict
from words_rewards.pipeline import Pipeline
from words_rewards.sourcing import AgenticWordSource, DirectWordSource
from words_rewards.storage import save_batch, slugify


def parse(argv):
    return build_parser().parse_args(argv)


# ------------------------------------------------------------------ config


def test_settings_reject_unknown_effort():
    with pytest.raises(ValueError, match="effort"):
        Settings(judge_effort="turbo").validate()


def test_settings_reject_out_of_range_threshold():
    with pytest.raises(ValueError, match="threshold"):
        Settings(pass_threshold=1.5).validate()


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("WR_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("WR_JUDGE_EFFORT", "max")
    monkeypatch.setenv("WR_PASS_THRESHOLD", "0.9")

    settings = Settings.from_env()

    assert settings.explainer_model == "claude-sonnet-5"
    assert settings.judge_effort == "max"
    assert settings.pass_threshold == 0.9


def test_cli_flags_override_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WR_MODEL", "claude-sonnet-5")
    args = parse(
        ["--judge-model", "claude-opus-5", "--threshold", "0.5", "--out", str(tmp_path)]
    )

    settings = settings_from_args(args)

    assert settings.judge_model == "claude-opus-5"
    assert settings.explainer_model == "claude-sonnet-5"
    assert settings.pass_threshold == 0.5
    assert settings.output_dir == tmp_path


# --------------------------------------------------------------------- cli


def test_word_and_count_are_mutually_exclusive(capsys):
    assert main(["--word", "serendipity", "--count", "3", "--mock"]) == 2
    assert "single word" in capsys.readouterr().err


def test_count_must_be_positive(capsys):
    assert main(["--count", "0", "--mock"]) == 2


def test_missing_credentials_exit_code(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    assert main(["--word", "serendipity"]) == 3
    assert "--mock" in capsys.readouterr().err


def test_mock_run_end_to_end(tmp_path, monkeypatch, offline_client, capsys):
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )

    exit_code = main(
        ["--mock", "--word", "serendipity", "--out", str(tmp_path), "--print-json"]
    )

    assert exit_code == 0
    files = sorted(tmp_path.glob("serendipity-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["word"] == "serendipity"
    assert 0.0 <= payload["score"] <= 1.0
    assert "serendipity" in capsys.readouterr().out


def test_fail_under_sets_a_non_zero_exit(tmp_path, monkeypatch, offline_client):
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )

    exit_code = main(
        [
            "--mock",
            "--word",
            "serendipity",
            "--out",
            str(tmp_path),
            "--fail-under",
            "0.99",
        ]
    )
    assert exit_code == 1


def test_mock_mode_never_uses_the_agentic_source(monkeypatch, offline_client):
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )
    from words_rewards.cli import build_pipeline

    pipeline = build_pipeline(parse(["--mock"]), Settings())
    assert isinstance(pipeline.source, DirectWordSource)


def test_agentic_source_is_the_default(monkeypatch, offline_client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )
    from words_rewards.cli import build_pipeline

    pipeline = build_pipeline(parse([]), Settings())
    assert isinstance(pipeline.source, AgenticWordSource)


def test_random_flag_skips_the_agent(monkeypatch, offline_client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )
    from words_rewards.cli import build_pipeline

    pipeline = build_pipeline(parse(["--random"]), Settings())
    assert isinstance(pipeline.source, DirectWordSource)


# ----------------------------------------------------------------- storage


@pytest.mark.parametrize(
    ("word", "slug"),
    [("de facto", "de-facto"), ("Serendipity!", "serendipity"), ("???", "word")],
)
def test_slugify(word, slug):
    assert slugify(word) == slug


def test_save_batch_reports_average_and_pass_rate(entry, settings, tmp_path):
    from tests.test_pipeline import ConstantJudge, StubSource
    from words_rewards.mock import TemplateExplainer

    records = [
        Pipeline(
            source=StubSource([entry]),
            explainer=TemplateExplainer(),
            judge=ConstantJudge(score),
            settings=settings,
        ).run_once()
        for score in (1.0, 0.4)
    ]

    path = save_batch(records, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["average_score"] == 0.7
    assert payload["pass_rate"] == 0.5
    assert len(payload["runs"]) == 2


def test_summarise_mentions_every_word(entry, settings):
    from tests.test_pipeline import ConstantJudge, StubSource
    from words_rewards.mock import TemplateExplainer

    record = Pipeline(
        source=StubSource([entry]),
        explainer=TemplateExplainer(),
        judge=ConstantJudge(0.75),
        settings=settings,
    ).run_once()

    text = summarise([record], settings)
    assert "serendipity" in text
    assert "PASS" in text
    assert "average score" in text


def test_judge_verdict_round_trips_through_json():
    verdict = JudgeVerdict(
        semantic_accuracy=0.5,
        sense_coverage=0.5,
        precision=0.5,
        correct_primary_sense=False,
        hallucinations=["invented an etymology"],
        missing_points=[],
        score=0.5,
        rationale="ok",
    )
    assert JudgeVerdict(**json.loads(verdict.model_dump_json())) == verdict
