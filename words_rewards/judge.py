"""Stage 3: LLM-as-a-judge scores the explanation against the dictionary."""

from __future__ import annotations

from typing import Protocol

import anthropic

from words_rewards.config import Settings
from words_rewards.llm import ADAPTIVE_THINKING, ensure_not_refused
from words_rewards.models import Explanation, JudgeVerdict, WordEntry

SYSTEM_PROMPT = """\
You are a strict lexicography grader. You are given a word, the dictionary.com
entry for it, and an explanation of that word written by another model that did
NOT see the entry. Grade the explanation against the entry.

The dictionary entry is the ground truth. The explanation does not have to match
its wording - only its meaning.

Scoring guide for the overall `score` (0.0 to 1.0):
- 1.0  every dictionary sense is conveyed correctly, nothing invented.
- 0.8  the primary sense is correct and well put; a secondary sense is thin or
       missing.
- 0.6  the primary sense is broadly right but vague, or one claim is wrong.
- 0.4  the right general area, but the core meaning is muddled.
- 0.2  mostly wrong, with an incidental overlap.
- 0.0  wrong word, wrong meaning, or an admission of not knowing.

Subtract for anything the entry contradicts. Do not reward length, confidence or
style. Grade only against the entry you were given; do not fall back on senses
you remember but that the entry does not list.\
"""


class Judge(Protocol):
    """Anything that can score an explanation against a dictionary entry."""

    def judge(self, entry: WordEntry, explanation: Explanation) -> JudgeVerdict:
        ...


def build_prompt(entry: WordEntry, explanation: Explanation) -> str:
    return "\n".join(
        [
            f"<word>{entry.word}</word>",
            "",
            "<dictionary_entry source=\"dictionary.com\">",
            entry.meaning,
            "</dictionary_entry>",
            "",
            f"<explanation model=\"{explanation.model}\">",
            explanation.text,
            "</explanation>",
            "",
            "Grade the explanation.",
        ]
    )


class ClaudeJudge:
    """Judge backed by the Claude API, using structured outputs."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or Settings()
        #: Updated after each call with the model the API actually served.
        self.model = self.settings.judge_model

    def judge(self, entry: WordEntry, explanation: Explanation) -> JudgeVerdict:
        message = self.client.messages.parse(
            model=self.settings.judge_model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            thinking=ADAPTIVE_THINKING,
            output_config={"effort": self.settings.judge_effort},
            output_format=JudgeVerdict,
            messages=[{"role": "user", "content": build_prompt(entry, explanation)}],
        )
        ensure_not_refused(message, "judge")
        self.model = message.model
        verdict = message.parsed_output
        if verdict is None:
            raise RuntimeError(
                f"judge returned no structured verdict "
                f"(stop_reason={message.stop_reason!r})"
            )
        return verdict
