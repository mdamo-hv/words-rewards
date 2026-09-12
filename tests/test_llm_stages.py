"""Tests for the Claude-backed stages, driven by a fake SDK client."""

from __future__ import annotations

import json
import types

import pytest

from words_rewards import explainer as explainer_module
from words_rewards import judge as judge_module
from words_rewards.config import Settings
from words_rewards.explainer import ClaudeExplainer
from words_rewards.judge import ClaudeJudge
from words_rewards.llm import ensure_not_refused, first_text
from words_rewards.models import Explanation, JudgeVerdict
from words_rewards.sourcing import AgenticWordSource, DictionaryError


def _text_block(text: str):
    return types.SimpleNamespace(type="text", text=text)


def _message(blocks, stop_reason="end_turn", model="claude-opus-5", parsed=None):
    return types.SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        model=model,
        parsed_output=parsed,
        stop_details=None,
    )


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


VERDICT = JudgeVerdict(
    semantic_accuracy=0.9,
    sense_coverage=0.8,
    precision=1.0,
    correct_primary_sense=True,
    hallucinations=[],
    missing_points=["the 'good fortune' sense"],
    score=0.85,
    rationale="Primary sense is right; one secondary sense is missing.",
)


# ---------------------------------------------------------------- explainer


def test_explainer_prompt_hides_the_meaning_by_default(entry):
    prompt = explainer_module.build_prompt(entry, show_meaning=False)

    assert entry.word in prompt
    assert "noun" in prompt
    assert entry.senses[0].definition not in prompt


def test_explainer_prompt_can_include_the_meaning(entry):
    prompt = explainer_module.build_prompt(entry, show_meaning=True)
    assert entry.senses[0].definition in prompt


def test_explainer_returns_the_model_text(entry):
    client = FakeClient(_message([_text_block("A happy accident.")]))
    explanation = ClaudeExplainer(client, Settings()).explain(entry)

    assert explanation.text == "A happy accident."
    assert explanation.word == entry.word
    assert explanation.saw_dictionary_meaning is False

    sent = client.messages.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"]["effort"] == "medium"


def test_explainer_raises_on_refusal(entry):
    client = FakeClient(_message([], stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="declined"):
        ClaudeExplainer(client, Settings()).explain(entry)


def test_first_text_requires_some_text():
    with pytest.raises(RuntimeError):
        first_text(_message([types.SimpleNamespace(type="thinking", text="")]))


def test_ensure_not_refused_passes_through_normal_stops():
    ensure_not_refused(_message([_text_block("hi")]), "explainer")


# -------------------------------------------------------------------- judge


def test_judge_prompt_contains_all_three_inputs(entry):
    explanation = Explanation(word=entry.word, text="A happy accident.", model="m")
    prompt = judge_module.build_prompt(entry, explanation)

    assert f"<word>{entry.word}</word>" in prompt
    assert entry.senses[0].definition in prompt
    assert "A happy accident." in prompt


def test_judge_returns_the_structured_verdict(entry):
    explanation = Explanation(word=entry.word, text="A happy accident.", model="m")
    client = FakeClient(_message([_text_block("{}")], parsed=VERDICT))
    judge = ClaudeJudge(client, Settings())

    verdict = judge.judge(entry, explanation)

    assert verdict.score == 0.85
    assert judge.model == "claude-opus-5"
    sent = client.messages.calls[0]
    assert sent["output_format"] is JudgeVerdict
    assert sent["output_config"]["effort"] == "high"


def test_judge_raises_when_no_verdict_is_returned(entry):
    explanation = Explanation(word=entry.word, text="A happy accident.", model="m")
    client = FakeClient(_message([_text_block("oops")], parsed=None))
    with pytest.raises(RuntimeError, match="no structured verdict"):
        ClaudeJudge(client, Settings()).judge(entry, explanation)


def test_verdict_rejects_scores_outside_the_unit_interval():
    with pytest.raises(ValueError):
        JudgeVerdict(**{**VERDICT.model_dump(), "score": 1.4})


def test_verdict_schema_is_generatable_by_the_sdk():
    from anthropic.lib._parse._transform import transform_schema
    from pydantic import TypeAdapter

    schema = transform_schema(TypeAdapter(JudgeVerdict).json_schema())
    assert schema["additionalProperties"] is False
    assert "score" in schema["required"]


# --------------------------------------------------------- sourcing agent


def test_sourcing_tools_capture_the_fetched_entry(offline_client):
    source = AgenticWordSource(client=None, dictionary=offline_client)
    captured: dict = {}
    list_wotd, fetch_entry = source._build_tools(captured)

    featured = json.loads(list_wotd.call({}))
    assert featured and featured[0]["word"]

    payload = json.loads(fetch_entry.call({"word": "serendipity"}))
    assert payload["word"] == "serendipity"
    assert captured["entry"].senses


def test_sourcing_tool_reports_lookup_failures_to_the_model(offline_client):
    source = AgenticWordSource(client=None, dictionary=offline_client)
    captured: dict = {}
    _, fetch_entry = source._build_tools(captured)

    payload = json.loads(fetch_entry.call({"word": "notarealword"}))
    assert "error" in payload
    assert captured == {}


def test_sourcing_tool_schemas_are_well_formed(offline_client):
    source = AgenticWordSource(client=None, dictionary=offline_client)
    _, fetch_entry = source._build_tools({})
    schema = fetch_entry.to_dict()

    assert schema["name"] == "fetch_dictionary_entry"
    assert schema["input_schema"]["required"] == ["word"]


def test_agent_that_never_fetches_is_an_error(offline_client, monkeypatch):
    source = AgenticWordSource(client=None, dictionary=offline_client)
    monkeypatch.setattr(
        source, "_build_tools", lambda captured: [], raising=True
    )

    class EmptyRunner:
        def __iter__(self):
            return iter(())

    fake_beta = types.SimpleNamespace(
        messages=types.SimpleNamespace(tool_runner=lambda **_: EmptyRunner())
    )
    source.client = types.SimpleNamespace(beta=fake_beta)

    with pytest.raises(DictionaryError, match="without fetching"):
        source.pull()
