"""NVIDIA Nemotron backend for the explainer, the judge and the word source.

Nemotron models are served from NVIDIA's OpenAI-compatible API catalogue, so
this module talks to them through the ``openai`` client rather than the
Anthropic SDK. The prompts and the rubric are shared with the Claude stages -
only the transport and the reasoning controls differ.
"""

from __future__ import annotations

import json
import logging
import os
import re

from pydantic import ValidationError

from words_rewards import explainer as explainer_prompts
from words_rewards import judge as judge_prompts
from words_rewards.config import Settings
from words_rewards.dictionary_source import DictionaryClient, DictionaryError
from words_rewards.llm import MissingCredentialsError
from words_rewards.models import Explanation, JudgeVerdict, WordEntry
from words_rewards.sourcing import (
    DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT as SOURCING_SYSTEM_PROMPT,
    as_openai_tools,
    build_dictionary_tools,
)

logger = logging.getLogger(__name__)

#: Environment variables searched for an NVIDIA API key, in order.
API_KEY_ENV_VARS = ("NVIDIA_API_KEY", "NEMOTRON_API_KEY")

#: Effort levels that turn Nemotron's reasoning mode on.
REASONING_EFFORTS = frozenset({"high", "xhigh", "max"})

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class NemotronError(RuntimeError):
    """Raised when a Nemotron call cannot be completed."""


def api_key() -> str | None:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def build_client(settings: Settings | None = None):
    """Create an OpenAI-compatible client pointed at NVIDIA's catalogue."""
    settings = settings or Settings()
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise NemotronError(
            "Nemotron models need the `openai` package: pip install openai"
        ) from exc

    key = api_key()
    if not key:
        raise MissingCredentialsError(
            "No NVIDIA credentials found. Export NVIDIA_API_KEY (get one at "
            "https://build.nvidia.com), or run with --mock."
        )
    return OpenAI(
        base_url=settings.nemotron_base_url,
        api_key=key,
        timeout=settings.request_timeout,
    )


def reasoning_directive(effort: str) -> str:
    """Nemotron switches reasoning with a `detailed thinking` system line."""
    return (
        "detailed thinking on"
        if effort in REASONING_EFFORTS
        else "detailed thinking off"
    )


def sampling(effort: str) -> dict[str, float]:
    """NVIDIA's recommended sampling for each reasoning mode."""
    if effort in REASONING_EFFORTS:
        return {"temperature": 0.6, "top_p": 0.95}
    return {"temperature": 0.0, "top_p": 1.0}


def strip_reasoning(text: str) -> str:
    """Drop the ``<think>`` block reasoning models prepend to their answer."""
    without_blocks = _THINK_BLOCK.sub("", text)
    if "</think>" in without_blocks:
        without_blocks = _OPEN_THINK.sub("", without_blocks)
    return without_blocks.strip()


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    cleaned = _CODE_FENCE.sub("", strip_reasoning(text)).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise NemotronError(f"no JSON object in the reply: {cleaned[:200]!r}")


def _message_text(choice) -> str:
    content = choice.message.content or ""
    if not content.strip():
        raise NemotronError(
            f"model returned no content (finish_reason={choice.finish_reason!r})"
        )
    return content


class NemotronExplainer:
    """Stage 2 on a Nemotron model."""

    def __init__(self, client, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or Settings()
        self.model = self.settings.explainer_model

    def explain(self, entry: WordEntry, *, show_meaning: bool = False) -> Explanation:
        effort = self.settings.explainer_effort
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": reasoning_directive(effort)},
                {"role": "system", "content": explainer_prompts.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": explainer_prompts.build_prompt(
                        entry, show_meaning=show_meaning
                    ),
                },
            ],
            **sampling(effort),
        )
        text = strip_reasoning(_message_text(response.choices[0]))
        if not text:
            raise NemotronError("model returned only reasoning, no explanation")
        return Explanation(
            word=entry.word,
            text=text,
            model=response.model or self.model,
            saw_dictionary_meaning=show_meaning,
        )


def verdict_schema() -> dict:
    schema = JudgeVerdict.model_json_schema()
    schema["additionalProperties"] = False
    return schema


class NemotronJudge:
    """Stage 3 on a Nemotron model.

    NVIDIA's catalogue serves many models, and not all of them accept
    ``response_format``. The judge asks for a JSON schema first and, if the
    endpoint rejects it, falls back to putting the schema in the prompt for the
    rest of the run.
    """

    max_attempts = 2

    def __init__(self, client, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or Settings()
        self.model = self.settings.judge_model
        self.schema_mode = True

    def _system_messages(self, effort: str) -> list[dict]:
        instructions = judge_prompts.SYSTEM_PROMPT
        if not self.schema_mode:
            instructions += (
                "\n\nReply with a single JSON object and nothing else. It must "
                "match this JSON schema:\n"
                + json.dumps(verdict_schema(), indent=2)
            )
        return [
            {"role": "system", "content": reasoning_directive(effort)},
            {"role": "system", "content": instructions},
        ]

    def _request(self, messages: list[dict], effort: str):
        kwargs = dict(
            model=self.model,
            max_tokens=self.settings.max_tokens,
            messages=messages,
            **sampling(effort),
        )
        if self.schema_mode:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_verdict",
                    "strict": True,
                    "schema": verdict_schema(),
                },
            }
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self.schema_mode and _is_unsupported_format(exc):
                logger.info(
                    "%s does not accept response_format; "
                    "asking for JSON in the prompt instead",
                    self.model,
                )
                self.schema_mode = False
                return None
            raise

    def judge(self, entry: WordEntry, explanation: Explanation) -> JudgeVerdict:
        effort = self.settings.judge_effort
        prompt = judge_prompts.build_prompt(entry, explanation)
        messages = self._system_messages(effort) + [
            {"role": "user", "content": prompt}
        ]

        last_error = ""
        for attempt in range(self.max_attempts):
            response = self._request(messages, effort)
            if response is None:  # schema rejected - rebuild with the fallback
                messages = self._system_messages(effort) + [
                    {"role": "user", "content": prompt}
                ]
                response = self._request(messages, effort)
            if response is None:  # pragma: no cover - defensive
                raise NemotronError("judge request could not be sent")

            reply = _message_text(response.choices[0])
            try:
                return JudgeVerdict(**extract_json(reply))
            except (NemotronError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "judge reply %d was not a usable verdict: %s",
                    attempt + 1,
                    last_error,
                )
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": (
                            "That was not a valid verdict object: "
                            f"{last_error}\nReply again with only the JSON "
                            "object required by the schema."
                        ),
                    },
                ]

        raise NemotronError(
            f"judge did not return a valid verdict after "
            f"{self.max_attempts} attempts: {last_error}"
        )


def _is_unsupported_format(exc: Exception) -> bool:
    """True when the endpoint rejected the request over ``response_format``."""
    status = getattr(exc, "status_code", None)
    if status not in (400, 404, 422):
        return False
    return "response_format" in str(exc) or "json_schema" in str(exc)


class NemotronWordSource:
    """Stage 1 on a Nemotron model, via OpenAI-style tool calling."""

    def __init__(
        self,
        client,
        dictionary: DictionaryClient,
        settings: Settings | None = None,
        instruction: str = DEFAULT_INSTRUCTION,
        max_iterations: int = 8,
    ) -> None:
        self.client = client
        self.dictionary = dictionary
        self.settings = settings or Settings()
        self.instruction = instruction
        self.max_iterations = max_iterations
        self.model = self.settings.sourcing_model
        self.deterministic = False
        self.transcript: list[str] = []

    def pull(self) -> WordEntry:
        captured: dict[str, WordEntry] = {}
        tools = build_dictionary_tools(self.dictionary, captured)
        by_name = {tool.to_dict()["name"]: tool for tool in tools}
        effort = self.settings.sourcing_effort

        messages: list[dict] = [
            {"role": "system", "content": reasoning_directive(effort)},
            {"role": "system", "content": SOURCING_SYSTEM_PROMPT},
            {"role": "user", "content": self.instruction},
        ]
        self.transcript = []

        for _ in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                messages=messages,
                tools=as_openai_tools(tools),
                tool_choice="auto",
                **sampling(effort),
            )
            reply = response.choices[0].message
            messages.append(reply.model_dump(exclude_none=True))

            if reply.content:
                self.transcript.append(strip_reasoning(reply.content))
            if not reply.tool_calls:
                break

            for call in reply.tool_calls:
                name = call.function.name
                arguments = call.function.arguments or "{}"
                self.transcript.append(f"[tool] {name}({arguments})")
                tool = by_name.get(name)
                if tool is None:
                    result = json.dumps({"error": f"no such tool: {name}"})
                else:
                    try:
                        result = tool.call(json.loads(arguments))
                    except json.JSONDecodeError as exc:
                        result = json.dumps({"error": f"bad arguments: {exc}"})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        entry = captured.get("entry")
        if entry is None:
            raise DictionaryError(
                "the sourcing agent finished without fetching a dictionary entry"
            )
        return entry
