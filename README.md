# Diabetes readmission prediction

Dataset: **Diabetes 130-US Hospitals, OpenML 4541** (101,766 stays). The positive class combines recorded readmissions `<30` and `>30`; `NO` is negative. Predictions use encounter/discharge information.

## Files

- `src/`: main Python implementation, including the prediction endpoint.
- `data/`: local raw dataset, original schema and provenance.
- `artifacts/model.joblib`: fitted complete Pipeline and fairness postprocessor.
- `results/notebook/`: saved notebook results, including the explicitly labelled restored Random Forest comparison.
- `results/optuna/`: parameters, metrics and fold records from the two historical studies.
- `results/reproduced/`: results reproduced by the scripts, including comparison artifacts.
- `examples/request.json`: raw-input example without the target.
- `diabetes.ipynb`: supplementary demonstrations and saved results, with restored comparison provenance.
- `optuna_results.db`: original studies, read-only in default commands.

## Installation

Validated locally with Python 3.13.7 and the versions in `requirements.txt`. These are observed working versions; a clean installation on another machine has not been tested.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this folder. The scripts use the supplied dataset without downloading it. For the optional notebook, a Jupyter frontend and `ipykernel==6.29.5` are separate from the script runtime.

## Reproduction

```sh
# Export historical notebook results and existing Optuna traces.
python -m src.export_results

# Reproduce data exploration.
python -m src.eda

# Inspect persisted best trials; no new tuning.
python -m src.tune

# Reconstruct the final Pipeline from the selected persisted parameters.
python -m src.train

# Evaluate the saved model and reproduce explanations.
python -m src.evaluate
python -m src.explain

# Reproduce the four-model comparison with patient-grouped five-fold CV.
python -m src.tune --compare
```

The comparison uses Logistic Regression, Random Forest (100 trees, random_state=42), LightGBM and CatBoost. Random Forest has been restored to the supplementary notebook and comparison. The historical database contains 50 completed LightGBM trials and 50 completed CatBoost trials. Default commands never launch new trials or modify that database.

Fresh comparison results include fold metrics, parameters and 20 fitted fold Pipelines. Historical notebook outputs and historical tuning traces are kept separate from newly reproduced results. Times depend on the execution environment; displayed historical tables may be rounded.

## Local prediction endpoint

The fitted artifact is supplied. To start the service:

```sh
python -m src.serve --host 127.0.0.1 --port 8000
```

In another terminal, from this folder:

```sh
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

The JSON body is `{"records": [{...}], "mitigated": false}`. Use the original feature names and raw values shown in the example. Do not include `readmitted` or create engineered features. The seven columns discarded by preparation are optional; other raw predictor columns are required. Numeric values must be JSON numbers; missing values can be null, except `time_in_hospital`, which must be positive.

The response contains `probabilities`, binary `predictions` and `mitigated`. The default decision threshold is 0.5. Set `mitigated` to `true` to use the saved equalized-odds postprocessor: it needs race information and changes binary decisions, not LightGBM probabilities. Its fixed random state makes the same ordered batch reproducible; randomized decisions may vary with batch composition.

The service binds to localhost. Invalid inputs return JSON errors; startup never trains or downloads a model.

## Methods and limitations

The Pipeline includes deterministic cleaning and feature creation through `FunctionTransformer`, followed by fitted imputation, encoding, scaling where applicable and the estimator. The preparation function is importable from `src.data`, so inference does not depend on notebook state.

Patients remain separate between fitting, validation and test; CV also groups by patient. Historical tuning used the complete development set before its later fitting/validation split. Validation is therefore **not independent of hyperparameter selection**. The test had already been evaluated in the source notebook; reproduction retains existing choices without selecting from its scores.

The complete cohort, including deceased patients, and the target covering readmission at either delay are retained. SHAP explanations are not causal. Fairness mitigation trades predictive performance against group disparities and does not prove universal fairness.

Course references (physical PDF pages): `Final Project Instruction.pdf`, pp. 1–3; `S5-full.pdf`, pp. 10–12, 32–39 and 48–55; `Session 7.pdf`, pp. 8–17; `Session 8.pdf`, pp. 18–22; `Session 9.pdf`, pp. 18–22. FunctionTransformer, portable serialization, file exports and the standard-library HTTP service are explicit implementation adaptations rather than APIs claimed to have been taught.

The final model reproduces recorded validation Brier/ECE and test ROC-AUC exactly. The four fresh baseline comparisons agree with the original report at its displayed metric precision, and raw-input and HTTP prediction parity were checked before packaging.

The report PDF, oral presentation and genuine Git history remain separate academic submission requirements. The report must disclose actual AI use and human contributions.
