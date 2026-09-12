"""Tests for the Nemotron backend, driven by a fake OpenAI-compatible client."""

from __future__ import annotations

import json

import pytest
from openai.types.chat import ChatCompletion

from words_rewards.config import Settings
from words_rewards.dictionary_source import DictionaryError
from words_rewards.llm import MissingCredentialsError
from words_rewards.models import Explanation, JudgeVerdict
from words_rewards.nemotron import (
    NemotronError,
    NemotronExplainer,
    NemotronJudge,
    NemotronWordSource,
    build_client,
    extract_json,
    reasoning_directive,
    sampling,
    strip_reasoning,
    verdict_schema,
)
from words_rewards.providers import (
    ANTHROPIC,
    NEMOTRON,
    describe_catalogue,
    provider_of,
    resolve_model,
)
from words_rewards.sourcing import as_openai_tools, build_dictionary_tools

VERDICT_JSON = {
    "semantic_accuracy": 0.9,
    "sense_coverage": 0.6,
    "precision": 1.0,
    "correct_primary_sense": True,
    "hallucinations": [],
    "missing_points": ["the 'good fortune' sense"],
    "score": 0.8,
    "rationale": "Primary sense right, one sense missing.",
}

MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


def completion(message: dict, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
        }
    )


def text_completion(text: str) -> ChatCompletion:
    return completion({"role": "assistant", "content": text})


def tool_call_completion(name: str, arguments: dict) -> ChatCompletion:
    return completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        finish_reason="tool_calls",
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot the messages: the caller keeps appending to the same list.
        self.calls.append(
            {**kwargs, "messages": [dict(m) for m in kwargs.get("messages", [])]}
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeOpenAI:
    def __init__(self, *responses):
        self.chat = type(
            "Chat", (), {"completions": FakeCompletions(responses)}
        )()

    @property
    def calls(self):
        return self.chat.completions.calls


class FakeBadRequest(Exception):
    status_code = 400

    def __str__(self):
        return "response_format is not supported for this model"


def nemotron_settings(**overrides) -> Settings:
    base = dict(
        sourcing_model=MODEL, explainer_model=MODEL, judge_model=MODEL
    )
    return Settings(**{**base, **overrides})


# ------------------------------------------------------------ model routing


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5", (ANTHROPIC, "claude-opus-5")),
        ("nemotron-super", (NEMOTRON, MODEL)),
        ("NEMOTRON-NANO", (NEMOTRON, "nvidia/nvidia-nemotron-nano-9b-v2")),
        ("nvidia/llama-3.1-nemotron-70b-instruct", (NEMOTRON, "nvidia/llama-3.1-nemotron-70b-instruct")),
        ("nemotron:my-org/custom", (NEMOTRON, "my-org/custom")),
        ("anthropic:claude-sonnet-5", (ANTHROPIC, "claude-sonnet-5")),
    ],
)
def test_model_routing(model, expected):
    assert resolve_model(model) == expected


def test_provider_of_defaults_to_anthropic():
    assert provider_of("claude-haiku-4-5") == ANTHROPIC


@pytest.mark.parametrize("model", ["", "   ", "nemotron:"])
def test_unusable_model_names_are_rejected(model):
    with pytest.raises(ValueError):
        resolve_model(model)


def test_catalogue_lists_every_alias():
    text = describe_catalogue()
    assert "nemotron-super" in text
    assert MODEL in text


# --------------------------------------------------------- reasoning helpers


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", "off"), ("medium", "off"), ("high", "on"), ("max", "on")],
)
def test_reasoning_directive_tracks_effort(effort, expected):
    assert reasoning_directive(effort) == f"detailed thinking {expected}"


def test_sampling_is_greedy_when_reasoning_is_off():
    assert sampling("low")["temperature"] == 0.0
    assert sampling("high")["temperature"] == 0.6


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>hmm</think>The answer.", "The answer."),
        ("reasoning first</think>The answer.", "The answer."),
        ("No reasoning here.", "No reasoning here."),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_extract_json_handles_fences_and_reasoning():
    raw = "<think>weighing</think>\n```json\n{\"score\": 0.5}\n```"
    assert extract_json(raw) == {"score": 0.5}


def test_extract_json_handles_surrounding_prose():
    assert extract_json('Here you go: {"score": 0.25} - done') == {"score": 0.25}


def test_extract_json_without_an_object_raises():
    with pytest.raises(NemotronError, match="no JSON object"):
        extract_json("I could not grade that.")


def test_verdict_schema_is_closed():
    schema = verdict_schema()
    assert schema["additionalProperties"] is False
    assert "score" in schema["required"]


# ----------------------------------------------------------------- explainer


def test_explainer_strips_reasoning_and_reports_the_served_model(entry):
    client = FakeOpenAI(text_completion("<think>recall</think>A happy accident."))
    explanation = NemotronExplainer(client, nemotron_settings()).explain(entry)

    assert explanation.text == "A happy accident."
    assert explanation.model == MODEL

    sent = client.calls[0]
    assert sent["model"] == MODEL
    assert sent["messages"][0]["content"] == "detailed thinking off"
    assert "You explain English words" in sent["messages"][1]["content"]


def test_explainer_turns_reasoning_on_at_high_effort(entry):
    client = FakeOpenAI(text_completion("A happy accident."))
    settings = nemotron_settings(explainer_effort="high")
    NemotronExplainer(client, settings).explain(entry)

    sent = client.calls[0]
    assert sent["messages"][0]["content"] == "detailed thinking on"
    assert sent["temperature"] == 0.6


def test_explainer_rejects_an_empty_reply(entry):
    client = FakeOpenAI(text_completion("   "))
    with pytest.raises(NemotronError, match="no content"):
        NemotronExplainer(client, nemotron_settings()).explain(entry)


def test_explainer_rejects_a_reply_that_is_only_reasoning(entry):
    client = FakeOpenAI(text_completion("<think>still thinking</think>"))
    with pytest.raises(NemotronError, match="only reasoning"):
        NemotronExplainer(client, nemotron_settings()).explain(entry)


# --------------------------------------------------------------------- judge


@pytest.fixture
def explanation(entry) -> Explanation:
    return Explanation(word=entry.word, text="A happy accident.", model=MODEL)


def test_judge_asks_for_a_json_schema_and_parses_the_verdict(entry, explanation):
    client = FakeOpenAI(text_completion(json.dumps(VERDICT_JSON)))
    verdict = NemotronJudge(client, nemotron_settings()).judge(entry, explanation)

    assert verdict.score == 0.8
    assert verdict.missing_points == ["the 'good fortune' sense"]
    sent = client.calls[0]
    assert sent["response_format"]["json_schema"]["name"] == "judge_verdict"


def test_judge_falls_back_when_the_endpoint_rejects_response_format(
    entry, explanation
):
    client = FakeOpenAI(
        FakeBadRequest(),
        text_completion("```json\n" + json.dumps(VERDICT_JSON) + "\n```"),
    )
    judge = NemotronJudge(client, nemotron_settings())

    verdict = judge.judge(entry, explanation)

    assert verdict.score == 0.8
    assert judge.schema_mode is False
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]
    assert "JSON schema" in client.calls[1]["messages"][1]["content"]


def test_judge_reraises_unrelated_api_errors(entry, explanation):
    class Boom(Exception):
        status_code = 500

    client = FakeOpenAI(Boom("upstream exploded"))
    with pytest.raises(Boom):
        NemotronJudge(client, nemotron_settings()).judge(entry, explanation)


def test_judge_retries_once_when_the_reply_is_not_a_verdict(entry, explanation):
    client = FakeOpenAI(
        text_completion("I think it was pretty good, honestly."),
        text_completion(json.dumps(VERDICT_JSON)),
    )
    verdict = NemotronJudge(client, nemotron_settings()).judge(entry, explanation)

    assert verdict.score == 0.8
    assert "Reply again" in client.calls[1]["messages"][-1]["content"]


def test_judge_gives_up_after_the_retry(entry, explanation):
    client = FakeOpenAI(
        text_completion("no idea"), text_completion("still no idea")
    )
    with pytest.raises(NemotronError, match="did not return a valid verdict"):
        NemotronJudge(client, nemotron_settings()).judge(entry, explanation)


def test_judge_rejects_an_out_of_range_score(entry, explanation):
    client = FakeOpenAI(
        text_completion(json.dumps({**VERDICT_JSON, "score": 7})),
        text_completion(json.dumps(VERDICT_JSON)),
    )
    verdict = NemotronJudge(client, nemotron_settings()).judge(entry, explanation)
    assert verdict.score == 0.8


# ------------------------------------------------------------- word sourcing


def test_tools_render_in_the_openai_shape(offline_client):
    tools = build_dictionary_tools(offline_client, {})
    specs = as_openai_tools(tools)

    assert [spec["function"]["name"] for spec in specs] == [
        "list_word_of_the_day",
        "fetch_dictionary_entry",
    ]
    assert specs[1]["function"]["parameters"]["required"] == ["word"]
    assert specs[1]["function"]["description"]


def test_word_source_returns_the_entry_the_tool_fetched(offline_client):
    client = FakeOpenAI(
        tool_call_completion("fetch_dictionary_entry", {"word": "serendipity"}),
        text_completion("<think>done</think>I picked serendipity."),
    )
    source = NemotronWordSource(client, offline_client, nemotron_settings())

    entry = source.pull()

    assert entry.word == "serendipity"
    assert entry.senses[0].definition.startswith("an aptitude for making")
    assert source.transcript[-1] == "I picked serendipity."
    assert client.calls[1]["messages"][-1]["role"] == "tool"


def test_word_source_reports_an_unknown_tool_back_to_the_model(offline_client):
    client = FakeOpenAI(
        tool_call_completion("look_up_thesaurus", {"word": "serendipity"}),
        text_completion("Sorry, I gave up."),
    )
    source = NemotronWordSource(client, offline_client, nemotron_settings())

    with pytest.raises(DictionaryError, match="without fetching"):
        source.pull()

    tool_reply = json.loads(client.calls[1]["messages"][-1]["content"])
    assert tool_reply["error"] == "no such tool: look_up_thesaurus"


def test_word_source_survives_malformed_tool_arguments(offline_client):
    broken = completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "fetch_dictionary_entry",
                        "arguments": "{not json",
                    },
                }
            ],
        },
        finish_reason="tool_calls",
    )
    client = FakeOpenAI(
        broken,
        tool_call_completion("fetch_dictionary_entry", {"word": "serendipity"}),
        text_completion("Second try worked."),
    )
    source = NemotronWordSource(client, offline_client, nemotron_settings())

    assert source.pull().word == "serendipity"


# --------------------------------------------------------------- credentials


def test_build_client_without_a_key(monkeypatch):
    for name in ("NVIDIA_API_KEY", "NEMOTRON_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MissingCredentialsError, match="NVIDIA_API_KEY"):
        build_client(Settings())


def test_build_client_uses_the_configured_base_url(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = build_client(Settings(nemotron_base_url="http://localhost:9/v1"))
    assert str(client.base_url).rstrip("/") == "http://localhost:9/v1"


# ---------------------------------------------------------------------- cli


def test_cli_routes_each_stage_to_its_own_backend(monkeypatch, offline_client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )
    from words_rewards.cli import build_parser, build_pipeline
    from words_rewards.judge import ClaudeJudge

    from words_rewards.cli import settings_from_args

    args = build_parser().parse_args(
        ["--explainer-model", "nemotron-super", "--judge-model", "claude-opus-5"]
    )
    pipeline = build_pipeline(args, settings_from_args(args))

    assert isinstance(pipeline.explainer, NemotronExplainer)
    assert pipeline.explainer.model == MODEL
    assert isinstance(pipeline.judge, ClaudeJudge)


def test_cli_uses_the_nemotron_agent_for_a_nemotron_sourcing_model(
    monkeypatch, offline_client
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setattr(
        "words_rewards.cli.DictionaryClient", lambda _settings: offline_client
    )
    from words_rewards.cli import build_parser, build_pipeline

    args = build_parser().parse_args([])
    pipeline = build_pipeline(args, nemotron_settings())

    assert isinstance(pipeline.source, NemotronWordSource)


def test_list_models_exits_cleanly(capsys):
    from words_rewards.cli import main

    assert main(["--list-models"]) == 0
    assert "nemotron-super" in capsys.readouterr().out


def test_verdict_from_a_nemotron_judge_is_the_same_shape():
    assert JudgeVerdict(**VERDICT_JSON).score == 0.8
