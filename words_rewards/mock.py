"""Offline stand-ins for the two LLM stages.

``--mock`` swaps these in so the full pipeline - including the live
dictionary.com scrape and the JSON output - can be exercised without API
credentials, and so the tests stay deterministic.
"""

from __future__ import annotations

import re

from words_rewards.models import Explanation, JudgeVerdict, WordEntry

_TOKEN = re.compile(r"[a-z']+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with which who whose you your this those these something
    someone thing""".split()
)


def _content_words(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


class TemplateExplainer:
    """Produces a paraphrase of the entry - useful wiring, not a real model."""

    model = "mock-explainer"

    def explain(self, entry: WordEntry, *, show_meaning: bool = False) -> Explanation:
        sense = entry.senses[0]
        pos = f", used as a {sense.part_of_speech}" if sense.part_of_speech else ""
        text = (
            f'"{entry.word}"{pos}, refers to {sense.definition.rstrip(".")}. '
            f"It shows up when people want to name exactly that idea."
        )
        return Explanation(
            word=entry.word,
            text=text,
            model=self.model,
            saw_dictionary_meaning=show_meaning,
        )


class OverlapJudge:
    """Scores by content-word overlap with the dictionary senses."""

    model = "mock-judge"

    def judge(self, entry: WordEntry, explanation: Explanation) -> JudgeVerdict:
        explanation_words = _content_words(explanation.text)

        sense_scores = []
        for sense in entry.senses:
            sense_words = _content_words(sense.definition)
            if not sense_words:
                continue
            sense_scores.append(
                len(sense_words & explanation_words) / len(sense_words)
            )
        if not sense_scores:
            sense_scores = [0.0]

        semantic_accuracy = round(sense_scores[0], 4)
        sense_coverage = round(sum(sense_scores) / len(sense_scores), 4)
        precision = 1.0 if explanation_words else 0.0
        score = round(
            0.6 * semantic_accuracy + 0.3 * sense_coverage + 0.1 * precision, 4
        )
        return JudgeVerdict(
            semantic_accuracy=semantic_accuracy,
            sense_coverage=sense_coverage,
            precision=precision,
            correct_primary_sense=semantic_accuracy >= 0.5,
            hallucinations=[],
            missing_points=[
                sense.definition
                for sense, overlap in zip(entry.senses, sense_scores)
                if overlap < 0.5
            ],
            score=score,
            rationale=(
                "Mock judge: score is the content-word overlap between the "
                "explanation and the dictionary senses, not a semantic judgement."
            ),
        )
