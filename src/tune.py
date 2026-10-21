"""Explicit Optuna resumption; never invoked by default training."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import optuna
from sklearn.model_selection import cross_validate
from .data import load_data, split_data, get_cv
from .models import make_pipeline, METRICS


def read_studies(database):
    """Read a temporary copy so the supplied database is never modified."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory) / "studies.db"
        shutil.copy2(database, copied)
        storage = "sqlite:///" + str(copied.resolve())
        result = {}
        for name in ["lightgbm", "catboost"]:
            study = optuna.load_study(study_name=name + "_tuning", storage=storage)
            result[name] = {"params": study.best_params, "value": study.best_value,
                            "fold_auc": study.best_trial.user_attrs["test_roc_auc"],
                            "trials": study.trials_dataframe()}
        return result


def objective(trial, name, data):
    if name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100, step=20),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    else:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 4, 8),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"])}
    scores = cross_validate(make_pipeline(data["X_development"].columns, name, params),
                            data["X_development"], data["y_development"], cv=get_cv(),
                            groups=data["groups_development"], scoring=METRICS, error_score="raise")
    for key, values in scores.items():
        trial.set_user_attr(key, values.tolist())
    return np.mean(scores["test_roc_auc"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--database", default="optuna_results.db")
    parser.add_argument("--work-database", default="artifacts/tuning_work.db")
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()
    destination = Path(args.work_database)
    if destination.resolve() == Path(args.database).resolve():
        raise ValueError("Use a separate work database to preserve the original trials.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(args.database, destination)
    raw, y = load_data(args.data)
    data = split_data(raw, y)
    for name in ["lightgbm", "catboost"]:
        study = optuna.create_study(study_name=name + "_tuning", direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            storage="sqlite:///" + str(destination.resolve()), load_if_exists=True)
        # Exact notebook policy: budget counts all recorded trials, not only COMPLETE.
        remaining = max(0, args.budget - len(study.trials))
        if remaining:
            study.optimize(lambda trial: objective(trial, name, data), n_trials=remaining)
        print(name, json.dumps(study.best_params), study.best_value)

if __name__ == "__main__":
    main()
