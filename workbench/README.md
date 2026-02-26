# Eval Workbench (JSON-first)

This Streamlit app lives in `workbench/main.py` and uses only local JSON files.

## Run

```bash
streamlit run workbench/main.py
```

## Data Sources

- Eval runs: `workbench/eval_runs.json`
- W&B local runs: `wandb/run-*/files/*`

## Tabs

- `Eval Analytics`: leaderboard, filters, comparisons, and failure analysis.
- `Create Eval Run`: form-based run creation with optional task JSON upload/paste.
- `W&B`: local W&B run summaries and scalar metric inspection.

## Eval JSON Format

Top-level shape:

```json
{
  "runs": []
}
```

Each run can include:

- metadata: `run_id`, `created_at`, `benchmark`, `model`, `provider`, `tags`
- config: `temperature`, `top_p`, `max_tokens`, `concurrency`
- optional run metrics: `success_rate`, `tokens_per_second`, `energy_joules`, `cost_usd`
- optional task rows (`tasks`) for automatic metric computation at app startup
