"""Command line entry point: ``python -m words_rewards``."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from words_rewards.config import EFFORT_LEVELS, Settings
from words_rewards.dictionary_source import DictionaryClient, DictionaryError
from words_rewards.explainer import ClaudeExplainer
from words_rewards.judge import ClaudeJudge
from words_rewards.llm import MissingCredentialsError, build_client
from words_rewards.mock import OverlapJudge, TemplateExplainer
from words_rewards.models import RunRecord
from words_rewards.pipeline import Pipeline
from words_rewards.sourcing import (
    DEFAULT_INSTRUCTION,
    AgenticWordSource,
    DirectWordSource,
)
from words_rewards.storage import save_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="words-rewards",
        description=(
            "Pull a word and its meaning from dictionary.com, have an LLM "
            "explain it, have an LLM-as-a-judge score the explanation from 0 "
            "to 1, and save the result as JSON."
        ),
    )

    selection = parser.add_argument_group("word selection")
    selection.add_argument(
        "--word",
        help="Score this exact word instead of letting the agent choose one.",
    )
    selection.add_argument(
        "--random",
        action="store_true",
        help=(
            "Skip the sourcing agent and pick a random Word of the Day "
            "(cheaper and fully deterministic given --seed)."
        ),
    )
    selection.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="What kind of word the sourcing agent should look for.",
    )
    selection.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many distinct words to score (default: 1).",
    )
    selection.add_argument(
        "--seed", type=int, help="Seed for random word selection."
    )

    models = parser.add_argument_group("models")
    models.add_argument("--sourcing-model", help="Model for the sourcing agent.")
    models.add_argument("--explainer-model", help="Model under test.")
    models.add_argument("--judge-model", help="Model acting as the judge.")
    models.add_argument(
        "--judge-effort",
        choices=EFFORT_LEVELS,
        help="Reasoning effort for the judge (default: high).",
    )
    models.add_argument(
        "--explainer-effort",
        choices=EFFORT_LEVELS,
        help="Reasoning effort for the explainer (default: medium).",
    )

    scoring = parser.add_argument_group("scoring and output")
    scoring.add_argument(
        "--threshold",
        type=float,
        help="Score at or above which a run counts as passing (default: 0.7).",
    )
    scoring.add_argument(
        "--fail-under",
        type=float,
        help="Exit non-zero if the average score falls below this value.",
    )
    scoring.add_argument(
        "--out",
        type=Path,
        help="Directory for the JSON results (default: ./results).",
    )
    scoring.add_argument(
        "--show-meaning-to-explainer",
        action="store_true",
        help=(
            "Give the explainer the dictionary meaning. This measures "
            "paraphrasing rather than knowledge, so it is off by default."
        ),
    )
    scoring.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Replace both LLM stages with deterministic stand-ins. Still "
            "scrapes dictionary.com; needs no API credentials."
        ),
    )
    scoring.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full result JSON to stdout as well as saving it.",
    )
    scoring.add_argument(
        "-v", "--verbose", action="store_true", help="Log each stage as it runs."
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    overrides = {
        "sourcing_model": args.sourcing_model,
        "explainer_model": args.explainer_model,
        "judge_model": args.judge_model,
        "judge_effort": args.judge_effort,
        "explainer_effort": args.explainer_effort,
        "pass_threshold": args.threshold,
        "output_dir": args.out,
    }
    applied = {key: value for key, value in overrides.items() if value is not None}
    settings = Settings(**{**base.__dict__, **applied})
    settings.validate()
    return settings


def build_pipeline(args: argparse.Namespace, settings: Settings) -> Pipeline:
    dictionary = DictionaryClient(settings)
    rng = random.Random(args.seed) if args.seed is not None else None

    if args.mock:
        explainer = TemplateExplainer()
        judge = OverlapJudge()
        client = None
    else:
        client = build_client(settings)
        explainer = ClaudeExplainer(client, settings)
        judge = ClaudeJudge(client, settings)

    if args.word:
        source = DirectWordSource(dictionary, word=args.word)
    elif args.random or client is None:
        source = DirectWordSource(dictionary, rng=rng)
    else:
        source = AgenticWordSource(
            client, dictionary, settings, instruction=args.instruction
        )

    return Pipeline(
        source=source,
        explainer=explainer,
        judge=judge,
        settings=settings,
        show_meaning_to_explainer=args.show_meaning_to_explainer,
    )


def summarise(records: list[RunRecord], settings: Settings) -> str:
    lines = []
    for record in records:
        mark = "PASS" if record.passed else "FAIL"
        lines.append(f"  {record.score:.2f}  {mark}  {record.word}")
        lines.append(f"        {record.verdict.rationale}")
    average = sum(record.score for record in records) / len(records)
    lines.append(
        f"\naverage score {average:.3f} over {len(records)} word(s) "
        f"(threshold {settings.pass_threshold:.2f})"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    if args.word and args.count > 1:
        print("--word scores a single word; drop --count", file=sys.stderr)
        return 2

    try:
        settings = settings_from_args(args)
        pipeline = build_pipeline(args, settings)
        records = pipeline.run(count=args.count, output_dir=settings.output_dir)
    except MissingCredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (DictionaryError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if len(records) > 1:
        batch_path = save_batch(records, settings.output_dir)
        print(f"batch written to {batch_path}")

    print(summarise(records, settings))
    print(f"results saved under {settings.output_dir}/")

    if args.print_json:
        payload = [record.to_json_dict() for record in records]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))

    if args.fail_under is not None:
        average = sum(record.score for record in records) / len(records)
        if average < args.fail_under:
            print(
                f"average score {average:.3f} is below --fail-under "
                f"{args.fail_under:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
