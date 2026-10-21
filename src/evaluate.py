"""Reproduce calibration and fairness diagnostics without fitting or model selection."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, confusion_matrix,
)

from src.data import load_data, sensitive_race, split_data


def calibration_table(y, probabilities):
    frame = pd.DataFrame({"predicted": probabilities, "observed": np.asarray(y)})
    frame["bin"] = pd.cut(frame["predicted"], bins=np.linspace(0, 1, 11), include_lowest=True)
    grouped = frame.groupby("bin", observed=True)
    table = grouped.agg(
        count=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean"),
    )
    table["gap"] = (table["predicted"] - table["observed"]).abs()
    table["weight"] = table["count"] / len(frame)
    ece = float((table["gap"] * table["weight"]).sum())
    return table, {"brier": float(brier_score_loss(y, probabilities)), "ece": ece}


def performance(y, prediction):
    return {
        "Accuracy": accuracy_score(y, prediction),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
    }


def by_group(y, prediction, sensitive):
    audit = pd.DataFrame({"race": np.asarray(sensitive), "actual": np.asarray(y), "predicted": prediction})
    rows = []
    for group, frame in audit.groupby("race"):
        tn, fp, fn, tp = confusion_matrix(frame["actual"], frame["predicted"], labels=[0, 1]).ravel()
        rows.append({
            "Group": group, "Stays": len(frame), "Prevalence": frame["actual"].mean(),
            "Selection rate": frame["predicted"].mean(), "TPR": tp / (tp + fn) if tp + fn else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan, "Precision": tp / (tp + fp) if tp + fp else np.nan,
            "False positives": int(fp), "False negatives": int(fn),
        })
    return pd.DataFrame(rows).set_index("Group")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output", default="results/reproduced")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    raw, y = load_data(args.data)
    parts = split_data(raw, y)
    model = bundle["pipeline"]
    probabilities = {part: model.predict_proba(parts[f"X_{part}"])[:, 1] for part in ["valid", "test"]}
    calibration_scores = {}
    for part in ["valid", "test"]:
        table, calibration_scores[part] = calibration_table(parts[f"y_{part}"], probabilities[part])
        table.to_csv(output / f"calibration_{part}_bins.csv")
        observed, predicted = calibration_curve(parts[f"y_{part}"], probabilities[part], n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(predicted, observed, "o-", label="LightGBM")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed readmission frequency", title=f"Calibration — {part}")
        ax.legend(); fig.tight_layout(); fig.savefig(output / f"calibration_{part}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probabilities["valid"], bins=np.linspace(0, 1, 11), edgecolor="black")
    ax.set(xlabel="Predicted readmission probability", ylabel="Number of stays", title="Predicted probabilities — validation")
    fig.tight_layout(); fig.savefig(output / "validation_probability_histogram.png", dpi=150); plt.close(fig)
    validation_groups = by_group(parts["y_valid"], (probabilities["valid"] >= .5).astype(int), sensitive_race(parts["X_valid"]))
    validation_groups.to_csv(output / "fairness_validation.csv")
    race = sensitive_race(parts["X_test"])
    before = (probabilities["test"] >= .5).astype(int)
    after = bundle["mitigator"].predict(parts["X_test"], sensitive_features=race, random_state=42)
    overall = pd.DataFrame({
        "Before": performance(parts["y_test"], before), "After": performance(parts["y_test"], after),
    }).T
    overall.to_csv(output / "performance_before_after.csv")
    tables = []
    for name, prediction in [("Before", before), ("After", after)]:
        table = by_group(parts["y_test"], prediction, race).reset_index()
        table.insert(0, "Version", name)
        tables.append(table)
    fairness = pd.concat(tables, ignore_index=True)
    fairness.to_csv(output / "metrics_by_group.csv", index=False)
    grouped = fairness.groupby("Version")[["TPR", "FPR"]]
    gaps = grouped.max() - grouped.min()
    gaps["Equalized odds gap"] = gaps.max(axis=1)
    gaps = gaps.reindex(["Before", "After"])
    gaps.to_csv(output / "fairness_gaps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["Before", "After"]):
        fairness[fairness["Version"] == name].set_index("Group")[["TPR", "FPR"]].plot.bar(ax=ax)
        ax.set(title=f"Test rates — {name.lower()}", ylabel="Rate", ylim=(0, 1)); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(output / "fairness_before_after.png", dpi=150); plt.close(fig)
    pd.DataFrame({
        "row_index": parts["X_test"].index, "race": race.to_numpy(), "actual": parts["y_test"].to_numpy(),
        "probability": probabilities["test"], "before": before, "after": after,
    }).to_csv(output / "test_predictions.csv.gz", index=False)
    summary = {
        "source": "Fresh script reproduction; fixed saved model, no selection or retraining",
        "calibration": calibration_scores, "test_roc_auc": float(roc_auc_score(parts["y_test"], probabilities["test"])),
        "performance": overall.to_dict(orient="index"), "fairness_gaps": gaps.to_dict(orient="index"),
        "validation_limitation": bundle["metadata"]["protocol_limitations"][0],
        "probabilities": "Baseline LightGBM probabilities; mitigation changes binary decisions only",
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
