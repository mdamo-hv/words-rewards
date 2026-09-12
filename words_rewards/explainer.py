"""Stage 2: the LLM under test explains the word."""

from __future__ import annotations

from typing import Protocol

import anthropic

from words_rewards.config import Settings
from words_rewards.llm import ADAPTIVE_THINKING, ensure_not_refused, first_text
from words_rewards.models import Explanation, WordEntry

SYSTEM_PROMPT = """\
You explain English words the way a good teacher would.

Rules:
- Explain what the word means in your own words: the core sense first, then any
  other senses in common use.
- Cover the part of speech and, where it helps, a short usage example.
- Be specific. Do not pad, and do not invent etymologies, senses or usages you
  are not confident about.
- If you do not know the word, say so plainly instead of guessing.
- Keep the whole answer under 180 words. Plain prose, no headings, no bullets.\
"""


class Explainer(Protocol):
    """Anything that can turn a word into an explanation."""

    def explain(self, entry: WordEntry, *, show_meaning: bool = False) -> Explanation:
        ...


def build_prompt(entry: WordEntry, *, show_meaning: bool) -> str:
    """Compose the explainer prompt.

    By default the explainer never sees the dictionary meaning - that is what
    makes the judge's score a measurement of the model's own knowledge rather
    than of its paraphrasing.
    """
    lines = [f'Explain the word "{entry.word}".']
    if entry.parts_of_speech:
        lines.append(
            "It is used as: " + ", ".join(entry.parts_of_speech) + "."
        )
    if show_meaning:
        lines.append(
            "For reference, dictionary.com defines it as:\n" + entry.meaning
        )
    return "\n\n".join(lines)


class ClaudeExplainer:
    """Explainer backed by the Claude API."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or Settings()

    def explain(self, entry: WordEntry, *, show_meaning: bool = False) -> Explanation:
        message = self.client.messages.create(
            model=self.settings.explainer_model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            thinking=ADAPTIVE_THINKING,
            output_config={"effort": self.settings.explainer_effort},
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(entry, show_meaning=show_meaning),
                }
            ],
        )
        ensure_not_refused(message, "explainer")
        return Explanation(
            word=entry.word,
            text=first_text(message),
            model=message.model,
            saw_dictionary_meaning=show_meaning,
        )
