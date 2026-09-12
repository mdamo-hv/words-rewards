"""Runtime configuration for the words-rewards pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5"

#: Effort levels accepted by ``output_config.effort``.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to know that is not per-run input."""

    # Model selection. The sourcing agent, the explainer and the judge are kept
    # separate on purpose: a judge should never be weaker than the model it
    # grades, but the explainer can be swapped to benchmark other models.
    sourcing_model: str = DEFAULT_MODEL
    explainer_model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL

    sourcing_effort: str = "low"
    explainer_effort: str = "medium"
    judge_effort: str = "high"

    max_tokens: int = 8000
    request_timeout: float = 180.0

    # Scraping.
    dictionary_base_url: str = "https://www.dictionary.com"
    http_timeout: float = 30.0
    http_retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # Scoring / output.
    pass_threshold: float = 0.7
    max_senses: int = 6
    output_dir: Path = field(default_factory=lambda: Path("results"))

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``WR_*`` environment variables."""
        default_model = _env_str("WR_MODEL", DEFAULT_MODEL)
        settings = cls(
            sourcing_model=_env_str("WR_SOURCING_MODEL", default_model),
            explainer_model=_env_str("WR_EXPLAINER_MODEL", default_model),
            judge_model=_env_str("WR_JUDGE_MODEL", default_model),
            sourcing_effort=_env_str("WR_SOURCING_EFFORT", "low"),
            explainer_effort=_env_str("WR_EXPLAINER_EFFORT", "medium"),
            judge_effort=_env_str("WR_JUDGE_EFFORT", "high"),
            max_tokens=_env_int("WR_MAX_TOKENS", 8000),
            request_timeout=_env_float("WR_REQUEST_TIMEOUT", 180.0),
            http_timeout=_env_float("WR_HTTP_TIMEOUT", 30.0),
            http_retries=_env_int("WR_HTTP_RETRIES", 3),
            pass_threshold=_env_float("WR_PASS_THRESHOLD", 0.7),
            max_senses=_env_int("WR_MAX_SENSES", 6),
            output_dir=Path(_env_str("WR_OUTPUT_DIR", "results")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        for name in ("sourcing_effort", "explainer_effort", "judge_effort"):
            value = getattr(self, name)
            if value not in EFFORT_LEVELS:
                raise ValueError(
                    f"{name}={value!r} is not one of {', '.join(EFFORT_LEVELS)}"
                )
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0 and 1")
        if self.max_senses < 1:
            raise ValueError("max_senses must be at least 1")
