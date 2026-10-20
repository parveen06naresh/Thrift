# Contributing to Thrift

Thanks for helping. Thrift's rule is simple: **every number in the README comes from a script
in this repo, on public data, and every quality knob carries a guarantee.** Contributions that
keep that true are very welcome.

## Set up

```bash
git clone https://github.com/parveen06naresh/thrift && cd thrift
pip install -e ".[dev]"
pytest -q && ruff check .
thrift-llm demo                 # offline gateway + dashboard, no API key
```

## Good first contributions

- **A provider preset**: one line in `PROVIDERS` (`thrift_llm/config.py`) plus a test in
  `tests/test_providers.py`. If the provider is not OpenAI-compatible, add an adapter like
  `thrift_llm/anthropic.py`.
- **Prices**: add or update rows in `thrift_llm/pricing.py` with the date and source page.
- **A better router scorer** (embeddings, a small classifier). It must plug into
  `Router`/`Ladder` and be re-certified; report the new numbers with `scripts/eval_ladder.py`.
- **A learned cache embedder** through `SemanticCache(embedder=...)`, certified with
  `scripts/eval_cache.py`.
- **Benchmarks on new public data** with per-question outputs from several models.

## Pull requests

- Keep each PR focused, with tests. CI runs ruff and pytest on Python 3.10 and 3.12.
- If a change moves a reported number, re-run the script and commit the new `results/*.json`.
- Never hand-edit results, and never claim a number the repo cannot reproduce.
