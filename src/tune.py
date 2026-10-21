"""Inspect existing tuning; optionally rerun model comparison or explicitly request new trials."""

import argparse
import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate

from src.data import grouped_cv, load_data, prepare_data, split_data
from src.models import METRICS, best_trial, make_pipeline


def suggested_params(trial, name):
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", .01, .2),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100, step=20),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", .01, .2),
        "depth": trial.suggest_int("depth", 4, 8),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--database", default="optuna_results.db", help="Historical database, opened read-only")
    parser.add_argument("--output", default="results/reproduced")
    parser.add_argument("--compare", action="store_true", help="Run 5-fold grouped comparison of the four baseline models")
    parser.add_argument("--trials", type=int, default=0, help="Number of NEW trials, opt-in only")
    parser.add_argument("--model", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--new-database", help="Separate writable SQLite database required for new trials")
    args = parser.parse_args()
    if args.trials < 0:
        parser.error("--trials must be nonnegative")
    if args.trials and (not args.new_database or Path(args.new_database).resolve() == Path(args.database).resolve()):
        parser.error("New trials require --new-database distinct from the historical --database")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {name: best_trial(args.database, f"{name}_tuning") for name in ["lightgbm", "catboost"]}
    (output / "persisted_best_trials.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not (args.compare or args.trials):
        return
    raw, y = load_data(args.data)
    parts = split_data(raw, y)
    columns = prepare_data(raw.iloc[:1]).columns
    if args.compare:
        rows, means = [], {}
        for name in ["logistic", "forest", "lightgbm", "catboost"]:
            print(f"Comparing {name} with 5-fold patient-grouped CV on development data", flush=True)
            pipeline = make_pipeline(columns, name)
            scores = cross_validate(
                pipeline, parts["X_development"], parts["y_development"],
                cv=grouped_cv(), groups=parts["groups_development"], scoring=METRICS, error_score="raise",
                return_estimator=True,
            )
            # Save the fitted fold pipelines as actual experiment artifacts, not fabricated historical artifacts.
            artifact_dir = output / "comparison_models"
            artifact_dir.mkdir(exist_ok=True)
            for fold, estimator in enumerate(scores.pop("estimator")):
                joblib.dump(estimator, artifact_dir / f"{name}_fold{fold}.joblib", compress=3)
            (artifact_dir / f"{name}_parameters.json").write_text(json.dumps(pipeline.named_steps["model"].get_params(), indent=2, default=str) + "\n")
            table = pd.DataFrame(scores)
            table.insert(0, "fold", range(len(table)))
            table.insert(0, "model", name)
            rows.append(table)
            means[name] = table.filter(like="test_").mean()
            pd.concat(rows, ignore_index=True).to_csv(output / "comparison_folds.csv", index=False)
            pd.DataFrame(means).T.to_csv(output / "comparison_means.csv", index_label="model")
    if args.trials:
        new_database = Path(args.new_database).resolve()
        new_database.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=f"{args.model}_new_tuning", direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42), storage=f"sqlite:///{new_database}", load_if_exists=True,
        )
        def objective(trial):
            pipeline = make_pipeline(columns, args.model, suggested_params(trial, args.model))
            scores = cross_validate(
                pipeline, parts["X_development"], parts["y_development"],
                cv=grouped_cv(), groups=parts["groups_development"], scoring=METRICS, error_score="raise",
                return_estimator=True,
            )
            trial_dir = output / "new_trial_models" / f"{args.model}_{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            for fold, estimator in enumerate(scores.pop("estimator")):
                joblib.dump(estimator, trial_dir / f"fold{fold}.joblib", compress=3)
            trial.set_user_attr("artifacts", str(trial_dir))
            for key, values in scores.items():
                trial.set_user_attr(key, values.tolist())
            return scores["test_roc_auc"].mean()
        study.optimize(objective, n_trials=args.trials)
        study.trials_dataframe().to_csv(output / f"{args.model}_new_trials.csv", index=False)


if __name__ == "__main__":
    main()
