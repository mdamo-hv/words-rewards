# words-rewards

An agentic pipeline that measures how well an LLM actually understands
dictionary words, and writes a score between 0 and 1 to a JSON file.

```
┌─────────────────────┐   word + meaning   ┌──────────────┐   explanation   ┌───────────┐
│ 1. sourcing agent   │ ─────────────────► │ 2. explainer │ ──────────────► │ 3. judge  │
│ dictionary.com      │                    │ LLM          │                 │ LLM       │
└─────────────────────┘                    └──────────────┘                 └─────┬─────┘
          │                                                                        │
          └──────────────────── dictionary meaning (ground truth) ─────────────────┤
                                                                                   ▼
                                                                     results/<word>-<ts>.json
                                                                     { "word": …, "score": 0.85, … }
```

1. **Sourcing agent** — Claude runs a tool loop over two dictionary.com tools
   (`list_word_of_the_day`, `fetch_dictionary_entry`) and decides which word to
   pull. It may never write a definition itself; a word only counts once the
   scrape returns an entry.
2. **Explainer** — the model under test is given the word (and its parts of
   speech) and explains it **without seeing the dictionary meaning**. That is
   deliberate: if the explainer sees the definition, the score measures
   paraphrasing rather than knowledge. Use `--show-meaning-to-explainer` for the
   other behaviour.
3. **LLM-as-a-judge** — a second call gets the word, the dictionary entry and
   the explanation, and returns a structured verdict (the Claude structured
   outputs API, so the shape is guaranteed): a 0.0–1.0 `score` plus the rubric
   sub-scores, hallucinations and missing senses behind it.
4. **Storage** — every run is written to `results/<word>-<timestamp>.json` and
   appended to `results/scores.jsonl`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e '.[dev]'
export ANTHROPIC_API_KEY=sk-ant-...      # see .env.example for optional settings
```

## Run

```bash
# Let the agent choose a word, then explain and score it
python -m words_rewards

# Score a specific word
python -m words_rewards --word serendipity

# Score five agent-chosen words on a theme, and print each stage as it runs
python -m words_rewards --count 5 --instruction "obscure legal terms" -v

# Skip the agent and pick a random Word of the Day (one fewer LLM call)
python -m words_rewards --random --seed 42

# Exercise the whole pipeline without an API key: the dictionary.com scrape is
# real, both LLM stages are deterministic stand-ins
python -m words_rewards --mock --word serendipity
```

Useful flags: `--judge-model` / `--explainer-model` (benchmark one model with
another as judge), `--judge-effort` / `--explainer-effort`, `--threshold`,
`--fail-under` (non-zero exit when the average score is too low), `--out`,
`--print-json`. `python -m words_rewards --help` lists them all.

## Output

`results/serendipity-20260912T000333Z.json`:

```json
{
  "word": "serendipity",
  "score": 0.85,
  "passed": true,
  "pass_threshold": 0.7,
  "verdict": {
    "semantic_accuracy": 0.9,
    "sense_coverage": 0.8,
    "precision": 1.0,
    "correct_primary_sense": true,
    "hallucinations": [],
    "missing_points": ["the 'good fortune; luck' sense"],
    "score": 0.85,
    "rationale": "The primary sense is stated precisely …"
  },
  "explanation": { "word": "serendipity", "text": "…", "model": "claude-opus-5" },
  "entry": { "word": "serendipity", "source_url": "…", "senses": [ … ] },
  "judge_model": "claude-opus-5",
  "created_at": "2026-09-12T00:03:33Z"
}
```

`results/scores.jsonl` keeps a one-line summary per run (`word`, `score`,
`passed`, models, timestamp) so a series of runs can be aggregated directly.

## Scoring rubric

The judge grades the explanation against the dictionary entry only — never
against what it happens to remember about the word:

| Score | Meaning |
| ----- | ------- |
| 1.0 | every dictionary sense conveyed correctly, nothing invented |
| 0.8 | primary sense correct and well put; a secondary sense thin or missing |
| 0.6 | primary sense broadly right but vague, or one claim wrong |
| 0.4 | right general area, core meaning muddled |
| 0.2 | mostly wrong, incidental overlap |
| 0.0 | wrong word, wrong meaning, or "I don't know" |

## Layout

| Path | What it does |
| ---- | ------------ |
| `words_rewards/dictionary_source.py` | dictionary.com scraping and parsing |
| `words_rewards/sourcing.py` | stage 1 — the agentic word source and its tools |
| `words_rewards/explainer.py` | stage 2 — the model under test |
| `words_rewards/judge.py` | stage 3 — LLM-as-a-judge, structured output |
| `words_rewards/pipeline.py` | orchestration and retries |
| `words_rewards/storage.py` | JSON result files and the `scores.jsonl` index |
| `words_rewards/mock.py` | deterministic stand-ins used by `--mock` and the tests |
| `words_rewards/cli.py` | argument parsing and the run summary |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs fully offline: the dictionary pages are saved as fixtures under
`tests/fixtures/`, and the Claude calls go through fake clients.

## Notes

- Scraping is best-effort against dictionary.com's current markup. If the page
  structure changes, `tests/test_dictionary_source.py` is where it will show up
  first; refresh the fixtures in `tests/fixtures/` alongside any parser change.
- Requests are made one at a time with a retry/backoff, to stay polite.
