"""Orchestration: word -> explanation -> score -> JSON."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from words_rewards.config import Settings
from words_rewards.dictionary_source import DictionaryError
from words_rewards.explainer import Explainer
from words_rewards.judge import Judge
from words_rewards.models import RunRecord, WordEntry
from words_rewards.sourcing import WordSource
from words_rewards.storage import save_run

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """Runs the three stages and persists the result."""

    source: WordSource
    explainer: Explainer
    judge: Judge
    settings: Settings = field(default_factory=Settings)
    show_meaning_to_explainer: bool = False

    def run_once(self, entry: WordEntry | None = None) -> RunRecord:
        """Score one word, pulling it first unless ``entry`` is supplied."""
        entry = entry or self.source.pull()
        logger.info("pulled %r from %s", entry.word, entry.source_url)

        explanation = self.explainer.explain(
            entry, show_meaning=self.show_meaning_to_explainer
        )
        logger.info("explained %r with %s", entry.word, explanation.model)

        verdict = self.judge.judge(entry, explanation)
        score = round(min(1.0, max(0.0, verdict.score)), 4)
        logger.info("judged %r: %.2f", entry.word, score)

        return RunRecord(
            word=entry.word,
            score=score,
            passed=score >= self.settings.pass_threshold,
            pass_threshold=self.settings.pass_threshold,
            verdict=verdict,
            explanation=explanation,
            entry=entry,
            judge_model=getattr(self.judge, "model", self.settings.judge_model),
        )

    def run(self, count: int = 1, output_dir: Path | None = None) -> list[RunRecord]:
        """Score ``count`` distinct words, saving each one as it completes."""
        if count < 1:
            raise ValueError("count must be at least 1")
        output_dir = output_dir or self.settings.output_dir

        records: list[RunRecord] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = count * 3

        while len(records) < count and attempts < max_attempts:
            attempts += 1
            try:
                entry = self.source.pull()
            except DictionaryError as exc:
                if getattr(self.source, "deterministic", False):
                    raise
                logger.warning("could not pull a word: %s", exc)
                continue

            key = entry.word.strip().lower()
            if key in seen:
                logger.info("skipping duplicate word %r", entry.word)
                continue
            seen.add(key)

            record = self.run_once(entry)
            path = save_run(record, output_dir)
            logger.info("saved %s", path)
            records.append(record)

        if not records:
            raise RuntimeError(
                f"no words could be scored after {attempts} attempts"
            )
        if len(records) < count:
            logger.warning(
                "only scored %d of %d requested words", len(records), count
            )
        return records
