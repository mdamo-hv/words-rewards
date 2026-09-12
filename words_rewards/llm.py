"""Shared helpers for talking to the Claude API."""

from __future__ import annotations

import os

import anthropic

from words_rewards.config import Settings

# Adaptive thinking: Claude decides how much to reason per request, and
# `output_config.effort` controls the depth/cost tradeoff per stage.
ADAPTIVE_THINKING: dict[str, str] = {"type": "adaptive"}


class MissingCredentialsError(RuntimeError):
    """Raised when no Anthropic credentials are configured."""


def build_client(settings: Settings | None = None) -> anthropic.Anthropic:
    """Create an Anthropic client from the ambient credentials.

    The SDK resolves ``ANTHROPIC_API_KEY``/``ANTHROPIC_AUTH_TOKEN`` (or an
    ``ant auth login`` profile) itself; this wrapper only turns the resulting
    ``TypeError`` into an actionable message.
    """
    settings = settings or Settings()
    if not (
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    ):
        raise MissingCredentialsError(
            "No Anthropic credentials found. Export ANTHROPIC_API_KEY, or run "
            "the pipeline with --mock to exercise it without the API."
        )
    return anthropic.Anthropic(timeout=settings.request_timeout)


def first_text(message: anthropic.types.Message) -> str:
    """Concatenate the text blocks of a response, skipping thinking blocks."""
    parts = [block.text for block in message.content if block.type == "text"]
    text = "\n".join(part.strip() for part in parts if part.strip())
    if not text:
        raise RuntimeError(
            f"model returned no text (stop_reason={message.stop_reason!r})"
        )
    return text


def ensure_not_refused(message: anthropic.types.Message, stage: str) -> None:
    """Fail loudly when a turn ended in a policy refusal."""
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        raise RuntimeError(f"{stage} request was declined (category={category!r})")
