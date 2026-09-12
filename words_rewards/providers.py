"""Which backend serves a given model id.

Every stage picks its own backend from the model name, so a Nemotron model can
be graded by a Claude judge (or the other way round) without extra flags.
"""

from __future__ import annotations

ANTHROPIC = "anthropic"
NEMOTRON = "nemotron"

#: Short names for Nemotron chat models on NVIDIA's API catalogue, verified
#: against ``GET /v1/models`` on 2026-09-12. NVIDIA retires ids (a retired one
#: answers ``410 Gone``), so ``--list-models`` also queries the live catalogue
#: rather than trusting this table. Any catalogue id works in full too - pass it
#: as-is, or prefix it with ``nemotron:`` if it does not look like a Nemotron
#: model.
NEMOTRON_ALIASES: dict[str, str] = {
    "nemotron-nano": "nvidia/nemotron-nano-3-30b-a3b",
    "nemotron-lightning": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nemotron-super": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
}

_PREFIXES = {"anthropic:": ANTHROPIC, "nemotron:": NEMOTRON}


def resolve_model(model: str) -> tuple[str, str]:
    """Return ``(provider, model_id)`` for a model name from the CLI or env.

    >>> resolve_model("claude-opus-5")
    ('anthropic', 'claude-opus-5')
    >>> resolve_model("nemotron-super")
    ('nemotron', 'nvidia/nemotron-3-super-120b-a12b')
    >>> resolve_model("nemotron:my-org/custom-build")
    ('nemotron', 'my-org/custom-build')
    """
    name = model.strip()
    if not name:
        raise ValueError("model must not be empty")

    for prefix, provider in _PREFIXES.items():
        if name.lower().startswith(prefix):
            remainder = name[len(prefix) :].strip()
            if not remainder:
                raise ValueError(f"{model!r} names a provider but no model")
            return provider, NEMOTRON_ALIASES.get(remainder.lower(), remainder)

    lowered = name.lower()
    if lowered in NEMOTRON_ALIASES:
        return NEMOTRON, NEMOTRON_ALIASES[lowered]
    if lowered.startswith("nvidia/") or "nemotron" in lowered:
        return NEMOTRON, name
    return ANTHROPIC, name


def provider_of(model: str) -> str:
    return resolve_model(model)[0]


def describe_aliases() -> str:
    """Human-readable list of the built-in Nemotron aliases."""
    width = max(len(alias) for alias in NEMOTRON_ALIASES)
    lines = ["Nemotron aliases (NVIDIA API catalogue ids):"]
    for alias, model_id in NEMOTRON_ALIASES.items():
        lines.append(f"  {alias.ljust(width)}  {model_id}")
    lines.append(
        "\nAny other catalogue id also works, e.g. "
        "--explainer-model nvidia/llama-3.1-nemotron-70b-instruct"
    )
    return "\n".join(lines)
