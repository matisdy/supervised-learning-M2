# Recorded hospital readmission

Supervised Learning & Model Engineering, Albert School.  
**Authors:** Matis Douay, Driss Haddadi and Rayan Faher Bonne.  
**Dataset:** Diabetes 130-US Hospitals, OpenML 4541.  
**Submission repository:** https://github.com/matisdy/diabetes-submission

The target is a recorded readmission at either delay (`<30` or `>30` = 1; `NO` = 0). `NO` does not establish that no readmission occurred elsewhere. This is a coursework proof of concept, not a clinically validated service.

## Installation

Run all commands from the repository root. Use Python 3.13; the pinned dependencies match the original Python 3.13.7 environment. LightGBM requires a compatible native OpenMP runtime.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

## Reproduce the results

```bash
python -m src.eda
python -m src.train
```

The training command loads the bundled dataset offline, recreates patient-grouped partitions, reads the retained Optuna studies, fits the selected LightGBM and fairness post-processor, and exports calibration, SHAP and fairness results under `results/reproduced/`. It saves `artifacts/model.joblib` and checks key metrics against the frozen notebook results. Generated outputs are overwritten.

The fitting/validation/test sets contain 65,146 / 16,467 / 20,153 stays. No patient crosses either split boundary. The single estimator Pipeline accepts 49 raw predictors: its first `PrepareEncounters` step performs fixed cleaning and feature calculations, followed by learned preprocessing and the classifier. Four model families were compared; the two strongest boosting candidates were tuned.

Default training does not repeat the four baseline CVs or run new Optuna trials. To repeat the baseline comparison or omit SHAP:

```bash
python -m src.train --baselines
python -m src.train --skip-shap
```

To export/resume the retained tuning studies:

```bash
python -m src.tune
```

The supplied database contains 50 completed trials per candidate. The default total budget is 50, so this starts no new trials. A separate `artifacts/tuning_work.db` is used; the original `optuna_results.db` is preserved. Increasing `--budget` intentionally runs further trials. Never mix changed data, targets or folds into the retained studies. See `python -m src.train --help` and `python -m src.tune --help` for options.

## Local prediction endpoint

Use the supplied model bundle, or rebuild it with `python -m src.train`:

```bash
python -m src.serve --host 127.0.0.1 --port 8000 --model artifacts/model.joblib
```

In a second terminal, call the endpoint with the supplied raw-encounter example:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @examples/request.json
```

`POST /predict` accepts `{"records": [raw_encounter, ...], "mitigation": false}`. Each record must contain all 49 raw predictors listed in `data/diabetes130us.json`, excluding `readmitted`. Numerical values must be finite integers; stay duration must be positive. Categorical values are strings or null. Unknown categories are supported by the base encoder. Input validation checks schema, not clinical plausibility.

The response contains the base probability and a threshold-0.5 decision. Setting `"mitigation": true` adds a race-dependent equalized-odds decision without changing the probability; the race group must have been seen during fitting. Optional `"random_seed": 42` reproduces the same whole-batch randomized output. Seeded single requests need not equal a seeded batch. The standard-library HTTP server is intended for a local demonstration.

## Files and report

- `src/`: loading, Pipeline construction, EDA, training, tuning, evaluation, SHAP and serving.
- `data/`: offline snapshot and schema/provenance metadata; see `data/README.md`.
- `optuna_results.db`: original LightGBM and CatBoost studies.
- `artifacts/model.joblib`: fitted Pipeline, fairness rules and metadata.
- `examples/request.json`: complete raw-encounter request for the endpoint.
- `results/reproduced/`: selected metrics, figures, comparisons and tuning histories.
- `reports/diabetes_report.pdf`: ten main pages and a contribution/AI-use appendix.
- `reports/latex/main.tex`, `reports/latex/images/`: editable report source.
- `presentation/deck.pptx`, `presentation/deck.pdf`: editable defense deck and PDF.

Compile the report with a standard LaTeX installation:

```bash
cd reports/latex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The retained cohort includes deaths; the validation subset participated in hyperparameter selection, and SHAP uses a mixed development sample. The report explains these limits and the accuracy/fairness trade-off. The three authors must confirm their actual contribution and full AI-use declarations, rehearse the seven-minute presentation and three-minute Q&A, and verify professor access before submitting the report PDF.
