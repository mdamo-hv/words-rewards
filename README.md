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

Each stage runs on either **Claude** or an **NVIDIA Nemotron** model — pick per
stage, so you can put Nemotron under test with Claude as the judge, or the other
way round. See [Models](#models).

1. **Sourcing agent** — the model runs a tool loop over two dictionary.com tools
   (`list_word_of_the_day`, `fetch_dictionary_entry`) and decides which word to
   pull. It may never write a definition itself; a word only counts once the
   scrape returns an entry.
2. **Explainer** — the model under test is given the word (and its parts of
   speech) and explains it **without seeing the dictionary meaning**. That is
   deliberate: if the explainer sees the definition, the score measures
   paraphrasing rather than knowledge. Use `--show-meaning-to-explainer` for the
   other behaviour.
3. **LLM-as-a-judge** — a second call gets the word, the dictionary entry and
   the explanation, and returns a structured verdict: a 0.0–1.0 `score` plus the
   rubric sub-scores, hallucinations and missing senses behind it. On Claude
   that uses the structured outputs API, so the shape is guaranteed; on Nemotron
   it asks for a JSON schema and repairs the reply if the endpoint cannot
   enforce one.
4. **Storage** — every run is written to `results/<word>-<timestamp>.json` and
   appended to `results/scores.jsonl`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e '.[dev]'

export ANTHROPIC_API_KEY=sk-ant-...      # for Claude stages
export NVIDIA_API_KEY=nvapi-...          # for Nemotron stages (build.nvidia.com)
```

You only need a key for the backends you actually use, and neither for `--mock`.
See `.env.example` for the optional settings.

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

Useful flags: `--judge-effort` / `--explainer-effort`, `--threshold`,
`--fail-under` (non-zero exit when the average score is too low), `--out`,
`--print-json`. `python -m words_rewards --help` lists them all.

## Models

`--sourcing-model`, `--explainer-model` and `--judge-model` each take a Claude
id or a Nemotron model; the backend follows from the name, so no extra flag is
needed.

```bash
# Put Nemotron under test, keep Claude as the judge
python -m words_rewards --explainer-model nemotron-super --judge-model claude-opus-5

# Run every stage on Nemotron (only NVIDIA_API_KEY needed)
WR_MODEL=nemotron-super python -m words_rewards

# Any NVIDIA catalogue id works in full
python -m words_rewards --explainer-model nvidia/llama-3.1-nemotron-70b-instruct
```

| Alias | NVIDIA catalogue id |
| ----- | ------------------- |
| `nemotron-nano` | `nvidia/nvidia-nemotron-nano-9b-v2` |
| `nemotron-super` | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| `nemotron-ultra` | `nvidia/llama-3.1-nemotron-ultra-253b-v1` |
| `nemotron-70b` | `nvidia/llama-3.1-nemotron-70b-instruct` |

`python -m words_rewards --list-models` prints the same table. A name is read as
Nemotron when it is one of these aliases, starts with `nvidia/`, or contains
`nemotron`; prefix it explicitly with `nemotron:` or `anthropic:` when that guess
would be wrong (for example `--judge-model nemotron:my-org/custom-build`).

**Pick a capable judge.** The judge is what turns the explanation into a number,
and small models grade badly — a 4B Nemotron scored a deliberately wrong
explanation ("serendipity: a type of Italian pasta") at 0.5 while its own
rationale said it was unrelated. Put the small model under test as the
*explainer* and keep a large model as the judge.

**Self-hosted endpoints.** `WR_NEMOTRON_BASE_URL` points the Nemotron backend at
any OpenAI-compatible server — a self-hosted NIM, vLLM, or `llama-server` — so a
local Nemotron build works without an NVIDIA key:

```bash
WR_NEMOTRON_BASE_URL=http://127.0.0.1:8899/v1 NVIDIA_API_KEY=local \
  python -m words_rewards --word serendipity \
    --explainer-model nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 \
    --judge-model nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
```

**When a model id goes stale.** NVIDIA retires catalogue ids, and a retired one
answers `410 Gone`. The run stops with the id named and a pointer to
`--list-models`, which queries `GET /v1/models` live (no API key needed) and
flags any built-in alias the catalogue has dropped.

**How effort maps onto each backend.** `--explainer-effort` / `--judge-effort`
take `low`…`max`. On Claude they set `output_config.effort` alongside adaptive
thinking. On Nemotron, `high` and above send `detailed thinking on` with NVIDIA's
recommended sampling (temperature 0.6, top-p 0.95) and anything lower sends
`detailed thinking off` with greedy decoding; the `<think>` block reasoning
models emit is stripped before the text is scored.

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
| `words_rewards/providers.py` | maps a model name to its backend |
| `words_rewards/sourcing.py` | stage 1 — the agentic word source and its tools |
| `words_rewards/explainer.py` | stage 2 — the model under test |
| `words_rewards/judge.py` | stage 3 — LLM-as-a-judge, structured output |
| `words_rewards/nemotron.py` | all three stages on NVIDIA Nemotron models |
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
`tests/fixtures/`, and both the Claude and the Nemotron calls go through fake
clients.

## Notes

- Scraping is best-effort against dictionary.com's current markup. If the page
  structure changes, `tests/test_dictionary_source.py` is where it will show up
  first; refresh the fixtures in `tests/fixtures/` alongside any parser change.
- Requests are made one at a time with a retry/backoff, to stay polite.
