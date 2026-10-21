"""Fit the notebook's fixed final model without rerunning Optuna."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from fairlearn.postprocessing import ThresholdOptimizer
from .data import load_data, split_data, get_cv, NUMERIC_COLUMNS, sensitive_groups
from .models import make_pipeline, METRICS
from .tune import read_studies
from .evaluate import evaluate
from .explain import explain
from .eda import run_eda

EXPECTED = {"test_roc_auc": 0.7069148710313543,
            "validation_brier": 0.21552468552847948,
            "validation_ece": 0.010763150222173357}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--database", default="optuna_results.db")
    parser.add_argument("--output", default="artifacts/model.joblib")
    parser.add_argument("--results", default="results/reproduced")
    parser.add_argument("--baselines", action="store_true", help="Explicitly rerun all four baseline CVs.")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    output, results = Path(args.output), Path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    raw, y = load_data(args.data)
    data = split_data(raw, y)
    run_eda(raw, y, results / "eda")
    studies = read_studies(args.database)
    tracking = results / "experiment_tracking"
    tracking.mkdir(exist_ok=True)
    best = {}
    for name, study in studies.items():
        study["trials"].to_csv(tracking / f"{name}_trials.csv", index=False)
        best[name] = {key: value for key, value in study.items() if key != "trials"}
    (tracking / "best_trials.json").write_text(json.dumps(best, indent=2))
    import matplotlib.pyplot as plt
    for name, study in studies.items():
        trials = study["trials"]
        plt.figure(figsize=(8, 4))
        plt.plot(trials.number, trials.value, "o", alpha=.5, label="Trial ROC-AUC")
        plt.plot(trials.number, trials.value.cummax(), label="Best ROC-AUC so far")
        plt.xlabel("Trial number"); plt.ylabel("Mean cross-validation ROC-AUC")
        plt.title(f"Optimisation history — {name}"); plt.legend(); plt.tight_layout()
        plt.savefig(tracking / f"{name}_history.png", dpi=160); plt.close()
    if args.baselines:
        means = {}
        baseline_parameters = {}
        for name in ["logistic", "forest", "lightgbm", "catboost"]:
            pipeline = make_pipeline(data["X_development"].columns, name)
            baseline_parameters[name] = pipeline["model"].get_params(deep=False)
            scores = pd.DataFrame(cross_validate(pipeline, data["X_development"], data["y_development"],
                cv=get_cv(), groups=data["groups_development"], scoring=METRICS, error_score="raise"))
            scores.to_csv(tracking / f"{name}_cv.csv", index=False)
            means[name] = scores.filter(like="test_").mean()
        pd.DataFrame(means).T.to_csv(tracking / "baseline_comparison.csv")
        (tracking / "baseline_parameters.json").write_text(json.dumps(baseline_parameters, indent=2))
    pipeline = make_pipeline(data["X_fit"].columns, "lightgbm", studies["lightgbm"]["params"])
    pipeline.fit(data["X_fit"], data["y_fit"])
    sensitive = sensitive_groups(data["X_valid"])
    fair_model = ThresholdOptimizer(estimator=pipeline, constraints="equalized_odds",
        objective="accuracy_score", predict_method="predict_proba", prefit=True)
    fair_model.fit(data["X_valid"], data["y_valid"], sensitive_features=sensitive)
    metadata = {
        "target": "Recorded readmission at any delay (<30 or >30), versus NO",
        "threshold": 0.5, "random_state": 42,
        "split": "First fold of GroupKFold(5, shuffle=True, random_state=42), applied twice",
        "raw_columns": raw.columns.tolist(),
        "raw_schema": json.loads(Path(args.data).with_name("diabetes130us.json").read_text())["schema"],
        "prepared_columns": pipeline["preprocessor"].feature_names_in_.tolist(),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": [c for c in pipeline["preprocessor"].feature_names_in_ if c not in NUMERIC_COLUMNS],
        "best_params": studies["lightgbm"]["params"],
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "optuna", "shap", "fairlearn", "joblib"]},
        "source_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "rows": {name: len(data[name]) for name in ["X_development", "X_fit", "X_valid", "X_test"]},
        "sensitive_groups": sorted(sensitive.unique().tolist()),
        "fixed_preparation": "Included in Pipeline step preparation; pass raw encounters directly.",
    }
    joblib.dump({"pipeline": pipeline, "fair_model": fair_model, "metadata": metadata}, output)
    (results / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    metrics = evaluate(pipeline, fair_model, data, results)
    local = [] if args.skip_shap else explain(pipeline, data["X_development"], results)
    parity = {name: {"actual": metrics[name], "notebook_expected": expected,
                     "matches_at_1e-12": bool(np.isclose(metrics[name], expected, rtol=0, atol=1e-12))}
              for name, expected in EXPECTED.items()}
    parity["fairness_before_eo"] = {"actual": metrics["fairness_gaps"]["Before"]["Equalized odds gap"],
        "notebook_expected": 0.20802271700099828,
        "matches_at_1e-12": bool(np.isclose(metrics["fairness_gaps"]["Before"]["Equalized odds gap"], 0.20802271700099828, rtol=0, atol=1e-12))}
    parity["fairness_after_eo"] = {"actual": metrics["fairness_gaps"]["After"]["Equalized odds gap"],
        "notebook_expected": 0.11088011088011096,
        "matches_at_1e-12": bool(np.isclose(metrics["fairness_gaps"]["After"]["Equalized odds gap"], 0.11088011088011096, rtol=0, atol=1e-12))}
    # The frozen notebook displays these four metrics rounded to three decimals.
    parity["performance_rounded_3dp"] = {
        "actual": {name: {key: round(value, 3) for key, value in values.items()}
                   for name, values in metrics["performance"].items()},
        "notebook_expected": {"Before": {"Accuracy": .648, "Precision": .623, "Recall": .569, "F1": .595},
                              "After": {"Accuracy": .642, "Precision": .607, "Recall": .599, "F1": .603}}}
    parity["performance_rounded_3dp"]["matches"] = (parity["performance_rounded_3dp"]["actual"] ==
                                                            parity["performance_rounded_3dp"]["notebook_expected"])
    parity["patient_overlap"] = 0
    parity["shap_local"] = local
    (results / "parity.json").write_text(json.dumps(parity, indent=2))
    if not all(parity[name]["matches_at_1e-12"] for name in EXPECTED) or not parity["performance_rounded_3dp"]["matches"] or not all(
            parity[name]["matches_at_1e-12"] for name in ["fairness_before_eo", "fairness_after_eo"]):
        raise RuntimeError("Metrics differ from frozen notebook; inspect parity.json before using this model.")
    print(json.dumps({"bundle": str(output), "metrics": metrics}, indent=2))

if __name__ == "__main__":
    main()
