# Public release allowlist

This file defines the intended public GitHub contents for ReviewRadar. It is a
release policy, not a request to delete local files. Existing research data,
models, reports, and presentation artifacts may remain in the local workspace
but should not be staged unless explicitly reviewed.

## Include

- `README.md`
- `RELEASE_ALLOWLIST.md`
- `.gitignore`
- `requirements.txt`
- `app.py`
- `src/`
- `tests/`
- `data/README.md`
- `docs/images/reviewradar-overview.png`
- `reports/` Markdown, JSON, and reviewed small CSV artifacts cited by the README
- `presentation/architecture.mmd`
- `presentation/data_flow.mmd`
- `presentation/ReviewRadar_Defense_Pack.md`
- `presentation/ReviewRadar_Project_Presentation.md`

## Exclude by default

- `.venv/`, caches, bytecode, IDE state, and local Streamlit secrets
- `data/raw/`
- `data/external/raw/` and `data/external/processed/`
- Raw or processed datasets, ZIP archives, pickle files, and generated CSVs
- `models/reviewradar_selected.joblib` and other model artifacts
- `reports/figures/` except deliberately reviewed release images
- `presentation/ReviewRadar_Project_Presentation.pptx`
- `presentation/ReviewRadar_Technical_Report.doc`
- The bundled ReDef and JIT-Smart source/data packages

## Provenance and licensing checks required before inclusion

- The ApacheJIT dataset has attribution information but its inspected Zenodo
  record did not expose clear reuse terms. Do not redistribute its raw CSV.
- The bundled ReDef and JIT-Smart materials are third-party research assets;
  their provenance and redistribution permissions have not been established.
- No project license is included. A license must be selected explicitly before
  publication; this policy does not choose one.

## Release verification

Before staging, verify that the release contains no secrets, credentials,
private data, raw datasets, model weights, or unrelated generated artifacts.
Run the test suite after selecting release files, then inspect the full staged
diff and file list before committing.
