# swingtrader

Personal, GitHub-native swing-trading research tool for scientifically scoring breakout setups
across weekly, daily, and (later) intraday timeframes. Analysis-only — no broker, no execution.

## What it does

- Ingests free/low-cost market data (yfinance) for a curated liquid universe
  (S&P 500 + Nasdaq 100 + liquid sector ETFs + benchmarks + watchlist)
- Computes objective technical and contextual features (trend, compression, volume,
  relative strength, regime)
- Models **Anchored VWAP** as a first-class feature family
- Detects base/pivot structure and tracks a breakout lifecycle state machine:
  `BASE -> ARMED -> TRIGGERED -> ACCEPTED -> CONFIRMED / FAILED / LATE / EXHAUSTED`
- Fits interpretable models (regularized logistic / ridge / quantile) tied to pre-registered
  outcome labels — **no arbitrary point-weighted scores**
- Reports rankings, score decomposition, calibration, and decile performance as static
  markdown/HTML under `reports/` and (optionally) GitHub Pages

## Non-goals

No broker integrations. No order management. No execution. No options pricing.
No rebalancing. No HFT.

## Project status

**Phases 1–5 complete — operational.** The full pipeline runs end-to-end:
ingest → features → state machine → labels → walk-forward model training →
calibrated scoring → ranked reports → score history → GitHub Pages.

See [docs/operator_guide.md](docs/operator_guide.md) for daily operating instructions
and [docs/release_checklist.md](docs/release_checklist.md) before the first Actions run.

## Quickstart (local)

```bash
# Install uv (https://docs.astral.sh/uv/)
# Then from the repo root:
uv sync --all-extras
uv run pytest -q
```

Without `uv`:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
pip install -e .[dev]
pytest -q
```

See [docs/run_locally.md](docs/run_locally.md) for full local-run instructions, including
how to seed the universe membership CSVs.

## GitHub Actions

- `ci.yml` — tests and lint on every PR and push to `main`
- `daily.yml` — scheduled 22:00 UTC Mon–Fri: ingest → score → commit artifacts
- `pages.yml` — deploys `docs/` to GitHub Pages on push to `main`

## Repository layout

```
config/       YAML configs: universe, features, AVWAP anchors, labels, regimes, scoring
data/         Rolling parquet datasets (daily/weekly) and generated feature/label/score tables
src/swingtrader/
              ingest/ features/ avwap/ bases/ states/ labels/ models/ scoring/
              validation/ journal/ reports/ utils/
reports/      Generated markdown/HTML — locally browsable and Pages-deployable
tests/        pytest suite
docs/         Architecture and methodology docs
.github/      Actions workflows
```

See [AGENTS.md](AGENTS.md) for guardrails that apply to future changes.
