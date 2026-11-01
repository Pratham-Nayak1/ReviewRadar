# ReviewRadar training summary

Selected model: **`random_forest`**

Selection used validation PR-AUC, with validation F1 and model name as deterministic tie-breakers. The final test split was not used.

The selected pipeline was refit on **87,357** train-plus-validation rows and saved to `models/reviewradar_selected.joblib`.

Features used:

- `la`
- `ld`
- `nf`
- `nd`
- `ns`
- `ent`
- `ndev`
- `age`
- `nuc`
- `aexp`
- `arexp`
- `asexp`

`commit_id`, `buggy`, `fix`, `year`, `author_date`, and `project` were not used as predictors. `project` was excluded deliberately to avoid repository-identity shortcuts; this first model is based only on change metrics.

Run with:

```powershell
python -m src.train
```
