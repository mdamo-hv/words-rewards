"""Data structures exchanged between the pipeline stages."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Sense(BaseModel):
    """A single numbered definition on a dictionary.com entry page."""

    part_of_speech: str = ""
    definition: str
    examples: list[str] = Field(default_factory=list)


class WordEntry(BaseModel):
    """Ground truth pulled from dictionary.com."""

    word: str
    source_url: str
    senses: list[Sense] = Field(default_factory=list)
    pronunciation: str = ""
    retrieved_at: datetime = Field(default_factory=utcnow)

    @property
    def meaning(self) -> str:
        """The senses rendered as the numbered block shown to the judge."""
        lines: list[str] = []
        for index, sense in enumerate(self.senses, start=1):
            pos = f"({sense.part_of_speech}) " if sense.part_of_speech else ""
            lines.append(f"{index}. {pos}{sense.definition}")
            for example in sense.examples:
                lines.append(f"   example: {example}")
        return "\n".join(lines)

    @property
    def parts_of_speech(self) -> list[str]:
        seen: list[str] = []
        for sense in self.senses:
            if sense.part_of_speech and sense.part_of_speech not in seen:
                seen.append(sense.part_of_speech)
        return seen


class Explanation(BaseModel):
    """What the explainer LLM produced for a word."""

    word: str
    text: str
    model: str
    saw_dictionary_meaning: bool = False


class JudgeVerdict(BaseModel):
    """Structured output schema for the LLM-as-a-judge.

    ``score`` is the headline 0.0-1.0 number written to the result file; the
    rubric fields exist so a low score can be explained without re-running the
    judge.
    """

    semantic_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Does the explanation convey the same meaning as the dictionary "
            "definition? 1.0 = fully equivalent, 0.0 = unrelated or wrong."
        ),
    )
    sense_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "What share of the dictionary senses the explanation accounts for. "
            "Covering the primary sense well earns at least 0.5."
        ),
    )
    precision: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Freedom from invented, overreaching or misleading claims. Every "
            "hallucinated detail lowers this."
        ),
    )
    correct_primary_sense: bool = Field(
        description="True if the explanation identifies the first dictionary sense."
    )
    hallucinations: list[str] = Field(
        description=(
            "Short quotes of claims in the explanation that the dictionary "
            "entry contradicts or does not support. Empty list if none."
        )
    )
    missing_points: list[str] = Field(
        description="Dictionary senses or nuances the explanation left out."
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overall 0.0-1.0 grade for the explanation, weighing semantic "
            "accuracy most heavily, then sense coverage, then precision."
        ),
    )
    rationale: str = Field(
        description="Two or three sentences justifying the score."
    )


class RunRecord(BaseModel):
    """One complete word -> explanation -> score run."""

    word: str
    score: float
    passed: bool
    pass_threshold: float
    verdict: JudgeVerdict
    explanation: Explanation
    entry: WordEntry
    judge_model: str
    created_at: datetime = Field(default_factory=utcnow)

    def to_json_dict(self) -> dict:
        """Serialise with ``word`` and ``score`` first, details after."""
        return self.model_dump(mode="json")
