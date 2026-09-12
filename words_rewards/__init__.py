"""Agentic pipeline that scores how well an LLM understands dictionary words.

The pipeline has three stages:

1. ``sourcing``  - an agent pulls a word and its meaning from dictionary.com
2. ``explainer`` - an LLM explains the word without seeing the dictionary meaning
3. ``judge``     - an LLM-as-a-judge scores the explanation against the meaning

The result of every run is persisted as JSON containing (at minimum) the word
and the score.
"""

from words_rewards.config import Settings
from words_rewards.models import Explanation, JudgeVerdict, RunRecord, WordEntry
from words_rewards.pipeline import Pipeline

__all__ = [
    "Explanation",
    "JudgeVerdict",
    "Pipeline",
    "RunRecord",
    "Settings",
    "WordEntry",
]

__version__ = "0.1.0"
