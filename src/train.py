"""Fit the notebook's final model and fairness postprocessor; never tune implicitly."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from fairlearn.postprocessing import ThresholdOptimizer
import joblib
import numpy as np

from src.data import load_data, prepare_data, sensitive_race, split_data, TARGET_MAPPING
from src.models import best_trial, make_pipeline


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--database", default="optuna_results.db")
    parser.add_argument("--output", default="artifacts/model.joblib")
    parser.add_argument("--results", default="results/reproduced")
    args = parser.parse_args()
    start = time.perf_counter()
    raw, y = load_data(args.data)
    parts = split_data(raw, y)
    trial = best_trial(args.database)
    pipeline = make_pipeline(prepare_data(raw.iloc[:1]).columns, params=trial["params"])
    print(f"Fitting LightGBM from persisted trial {trial['number']} on {len(parts['X_fit'])} stays.", flush=True)
    pipeline.fit(parts["X_fit"], parts["y_fit"])
    mitigator = ThresholdOptimizer(
        estimator=pipeline, constraints="equalized_odds", objective="accuracy_score",
        predict_method="predict_proba", prefit=True,
    )
    mitigator.fit(parts["X_valid"], parts["y_valid"], sensitive_features=sensitive_race(parts["X_valid"]))
    snapshot = json.loads(Path(args.data).with_suffix("").with_suffix(".json").read_text())
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Fresh reproduction of supplied diabetes.ipynb; original protocol retained",
        "target": TARGET_MAPPING, "prediction_time": "At discharge",
        "split_counts": {part: len(parts[f"X_{part}"]) for part in ["development", "fit", "valid", "test"]},
        "split_patients": {part: parts[f"X_{part}"]["patient_nbr"].nunique() for part in ["development", "fit", "valid", "test"]},
        "selected_study": trial,
        "versions": {package: version(package) for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "fairlearn", "optuna", "joblib", "shap"]},
        "source_hashes": {str(path): sha256(path) for path in [args.data, args.database, "diabetes.ipynb"]},
        "protocol_limitations": [
            "Validation belongs to development previously used by Optuna; it is not independent of hyperparameter selection.",
            "Test was already examined in the source notebook; no new choices are made using its scores.",
            "Cohort and binary target include all stays and readmissions regardless of delay, including deceased patients.",
            "Fairness mitigation requires race at prediction time and changes decisions, not baseline probabilities.",
        ],
        "fit_elapsed_seconds": time.perf_counter() - start,
    }
    bundle = {
        "pipeline": pipeline, "mitigator": mitigator, "input_columns": list(raw.columns),
        "schema": {column: snapshot["schema"][column] for column in raw.columns}, "metadata": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Exactly the notebook's raw-vs-prepared check, with an importable function.
    examples = parts["X_test"].iloc[:3]
    raw_probabilities = pipeline.predict_proba(examples)[:, 1]
    prepared_probabilities = pipeline.predict_proba(prepare_data(examples))[:, 1]
    same = bool(np.allclose(raw_probabilities, prepared_probabilities))
    if not same:
        raise AssertionError("Raw and prepared prediction paths differ.")
    (results / "raw_input_check.json").write_text(json.dumps({
        "rows": examples.index.tolist(), "probabilities": raw_probabilities.tolist(),
        "same_predictions": same,
    }, indent=2) + "\n")
    print(f"Saved {output}; raw-input check={same}; elapsed={metadata['fit_elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
