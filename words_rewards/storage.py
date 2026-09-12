"""Persistence: every run ends as a JSON file holding the word and the score."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from words_rewards.models import RunRecord

_SLUG = re.compile(r"[^a-z0-9]+")

#: Append-only summary of every run, one JSON object per line.
INDEX_FILENAME = "scores.jsonl"


def slugify(word: str) -> str:
    slug = _SLUG.sub("-", word.strip().lower()).strip("-")
    return slug or "word"


def result_path(record: RunRecord, output_dir: Path) -> Path:
    stamp = record.created_at.strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"{slugify(record.word)}-{stamp}.json"


def save_run(record: RunRecord, output_dir: Path, path: Path | None = None) -> Path:
    """Write one run to ``output_dir`` and append it to the index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = path or result_path(record, output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    append_to_index(record, output_dir)
    return target


def append_to_index(record: RunRecord, output_dir: Path) -> Path:
    index = output_dir / INDEX_FILENAME
    summary = {
        "word": record.word,
        "score": record.score,
        "passed": record.passed,
        "judge_model": record.judge_model,
        "explainer_model": record.explanation.model,
        "created_at": record.created_at.isoformat(),
    }
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return index


def load_index(output_dir: Path) -> list[dict]:
    index = output_dir / INDEX_FILENAME
    if not index.exists():
        return []
    rows = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_batch(records: list[RunRecord], output_dir: Path) -> Path:
    """Write a combined file for a multi-word run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    target = output_dir / f"batch-{stamp}.json"
    scores = [record.score for record in records]
    payload = {
        "scores": [
            {"word": record.word, "score": record.score, "passed": record.passed}
            for record in records
        ],
        "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "pass_rate": (
            round(sum(1 for r in records if r.passed) / len(records), 4)
            if records
            else 0.0
        ),
        "runs": [record.to_json_dict() for record in records],
    }
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target
