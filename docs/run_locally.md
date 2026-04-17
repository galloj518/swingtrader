# Running locally

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run pytest -q
```

Without uv (plain venv + pip):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash; use bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest -q
```

## Seeding the universe

The default watchlist + sector ETFs + benchmarks is enough to exercise the pipeline out
of the box. To expand to the full S&P 500 / Nasdaq 100:

1. Create `data/reference/sp500.csv` with a `symbol` column listing members.
2. Create `data/reference/nasdaq100.csv` the same way.

Both files are loaded by `src/swingtrader/ingest/universe.py`. A future
`scripts/refresh_universe.py` will automate this; until then, drop the CSVs manually.

## Running the pipeline

```bash
# Full daily run (ingest + features + states + labels + snapshot):
uv run python -m swingtrader.pipelines.daily_run

# Score + rank + render reports:
uv run python -m swingtrader.pipelines.score_run

# Retrain models then score (run weekly or after config changes):
uv run python -m swingtrader.pipelines.score_run --train --skip-oos

# Specific date:
uv run python -m swingtrader.pipelines.daily_run 2025-03-14
uv run python -m swingtrader.pipelines.score_run 2025-03-14
```

See [docs/operator_guide.md](docs/operator_guide.md) for the full daily operating reference,
including how to handle ticker failures and known limitations.
