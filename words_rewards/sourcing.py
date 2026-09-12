"""Stage 1: pull a word and its meaning from dictionary.com.

Two implementations are available:

``DirectWordSource``
    Deterministic - looks up a word you name, or picks one of the featured
    Word of the Day words at random.

``AgenticWordSource``
    An agent loop: Claude is handed the dictionary tools and decides which word
    to pull, which is what you want when the selection itself needs judgement
    ("something a language learner would trip over", "a legal term", ...). The
    Nemotron equivalent is ``nemotron.NemotronWordSource``; both drive the same
    tools built by :func:`build_dictionary_tools`.
"""

from __future__ import annotations

import json
import random
from typing import Protocol

import anthropic
from anthropic import beta_tool
from anthropic.lib.tools import BetaFunctionTool

from words_rewards.config import Settings
from words_rewards.dictionary_source import DictionaryClient, DictionaryError
from words_rewards.llm import ADAPTIVE_THINKING
from words_rewards.models import WordEntry

SYSTEM_PROMPT = """\
You source vocabulary for a benchmark that measures how well a language model
understands dictionary words.

Work only with dictionary.com through the tools you are given. Never write a
definition yourself - a word only counts once `fetch_dictionary_entry` has
returned its entry.

Procedure:
1. Choose a candidate word that fits the request. `list_word_of_the_day` gives
   you dictionary.com's featured words if you need inspiration.
2. Call `fetch_dictionary_entry` on it.
3. If the lookup fails or the entry has no usable definition, pick a different
   word and try again. Give up after four failed lookups.
4. Once you have a good entry, stop and reply with just the word you settled on
   and one sentence on why you picked it.\
"""

DEFAULT_INSTRUCTION = (
    "Pull one interesting English word and its meaning. Prefer a word that is "
    "real but not everyday vocabulary, so the benchmark is not trivially easy."
)


def build_dictionary_tools(
    dictionary: DictionaryClient, captured: dict[str, WordEntry]
) -> list[BetaFunctionTool]:
    """Build the two dictionary.com tools the sourcing agent drives.

    The entry a successful lookup returns is stashed in ``captured["entry"]``,
    so the caller ends up with the scraped record rather than with whatever the
    model says about it. The tools carry a JSON schema and are callable, which
    is all either agent loop needs.
    """

    @beta_tool
    def list_word_of_the_day() -> str:
        """List the words dictionary.com currently features as Word of the Day.

        Returns a JSON array of objects with `word`, `date`,
        `part_of_speech` and a one-line `short_definition`.
        """
        try:
            featured = dictionary.fetch_word_of_the_day()
        except DictionaryError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(
            [
                {
                    "word": item.word,
                    "date": item.date,
                    "part_of_speech": item.part_of_speech,
                    "short_definition": item.short_definition,
                }
                for item in featured
            ]
        )

    @beta_tool
    def fetch_dictionary_entry(word: str) -> str:
        """Pull a word and its full meaning from dictionary.com.

        Args:
            word: The headword to look up, for example "serendipity".
        """
        try:
            entry = dictionary.fetch_entry(word)
        except DictionaryError as exc:
            return json.dumps({"error": str(exc), "word": word})
        captured["entry"] = entry
        return json.dumps(
            {
                "word": entry.word,
                "source_url": entry.source_url,
                "pronunciation": entry.pronunciation,
                "senses": [sense.model_dump() for sense in entry.senses],
            }
        )

    return [list_word_of_the_day, fetch_dictionary_entry]


def as_openai_tools(tools: list[BetaFunctionTool]) -> list[dict]:
    """Render the same tools in the OpenAI-compatible shape Nemotron expects."""
    specs = []
    for tool in tools:
        definition = tool.to_dict()
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": definition["name"],
                    "description": definition["description"],
                    "parameters": definition["input_schema"],
                },
            }
        )
    return specs


class WordSource(Protocol):
    """Anything that can produce a word plus its dictionary meaning."""

    #: True when the source always yields the same word, so retrying a failed
    #: pull cannot help.
    deterministic: bool

    def pull(self) -> WordEntry:
        ...


class DirectWordSource:
    """Look up a named word, or a random featured word."""

    def __init__(
        self,
        dictionary: DictionaryClient,
        word: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.dictionary = dictionary
        self.word = word
        self.rng = rng
        self.deterministic = word is not None

    def pull(self) -> WordEntry:
        word = self.word or self.dictionary.random_word(self.rng)
        return self.dictionary.fetch_entry(word)


class AgenticWordSource:
    """Let Claude choose the word, using dictionary.com tools."""

    def __init__(
        self,
        client: anthropic.Anthropic,
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
        self.deterministic = False
        self.transcript: list[str] = []

    def _build_tools(self, captured: dict[str, WordEntry]):
        return build_dictionary_tools(self.dictionary, captured)

    def pull(self) -> WordEntry:
        captured: dict[str, WordEntry] = {}
        runner = self.client.beta.messages.tool_runner(
            model=self.settings.sourcing_model,
            max_tokens=self.settings.max_tokens,
            max_iterations=self.max_iterations,
            system=SYSTEM_PROMPT,
            thinking=ADAPTIVE_THINKING,
            output_config={"effort": self.settings.sourcing_effort},
            tools=self._build_tools(captured),
            messages=[{"role": "user", "content": self.instruction}],
        )

        self.transcript = []
        for message in runner:
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    self.transcript.append(block.text.strip())
                elif block.type == "tool_use":
                    self.transcript.append(f"[tool] {block.name}({block.input})")

        entry = captured.get("entry")
        if entry is None:
            raise DictionaryError(
                "the sourcing agent finished without fetching a dictionary entry"
            )
        return entry
