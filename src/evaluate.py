"""Calibration, group error rates and final held-out evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from .data import sensitive_groups
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix)


def metrics_by_group(y, predictions, sensitive):
    rows = []
    audit = pd.DataFrame({"race": np.asarray(sensitive), "actual": np.asarray(y),
                          "predicted": np.asarray(predictions)})
    for group, data in audit.groupby("race"):
        tn, fp, fn, tp = confusion_matrix(data.actual, data.predicted, labels=[0, 1]).ravel()
        rows.append({"Group": group, "Stays": len(data), "Prevalence": data.actual.mean(),
                     "Selection rate": data.predicted.mean(),
                     "TPR": tp / (tp + fn) if tp + fn else np.nan,
                     "FPR": fp / (fp + tn) if fp + tn else np.nan,
                     "Precision": tp / (tp + fp) if tp + fp else np.nan,
                     "False positives": int(fp), "False negatives": int(fn)})
    return pd.DataFrame(rows).set_index("Group")


def evaluate(pipeline, fair_model, data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    valid_proba = pipeline.predict_proba(data["X_valid"])[:, 1]
    calibration = pd.DataFrame({"predicted": valid_proba,
                                "observed": data["y_valid"].to_numpy()})
    calibration["bin"] = pd.cut(calibration.predicted, np.linspace(0, 1, 11), include_lowest=True)
    grouped = calibration.groupby("bin", observed=True)
    calibration_table = grouped.agg(count=("observed", "size"), predicted=("predicted", "mean"),
                                    observed=("observed", "mean"))
    ece = ((calibration_table.predicted - calibration_table.observed).abs()
           * calibration_table["count"] / len(calibration)).sum()
    calibration_table.to_csv(output / "calibration_bins.csv")
    observed, predicted = calibration_curve(data["y_valid"], valid_proba, n_bins=10, strategy="uniform")
    plt.plot(predicted, observed, "o-", label="LightGBM")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed readmission frequency")
    plt.title("Calibration curve — validation set"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "calibration_curve.png", dpi=160); plt.close()
    plt.hist(valid_proba, bins=np.linspace(0, 1, 11), edgecolor="black")
    plt.xlabel("Predicted readmission probability"); plt.ylabel("Number of stays")
    plt.title("Predicted probabilities — validation set"); plt.tight_layout()
    plt.savefig(output / "probability_histogram.png", dpi=160); plt.close()
    sensitive_valid = sensitive_groups(data["X_valid"])
    metrics_by_group(data["y_valid"], (valid_proba >= .5).astype(int), sensitive_valid).to_csv(
        output / "validation_group_metrics.csv")
    X_test, y_test = data["X_test"], data["y_test"]
    sensitive = sensitive_groups(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    predictions = {"Before": (proba >= .5).astype(int),
                   "After": fair_model.predict(X_test, sensitive_features=sensitive, random_state=42)}
    performance, groups, gaps = {}, [], {}
    for name, prediction in predictions.items():
        performance[name] = {
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1": f1_score(y_test, prediction, zero_division=0)}
        group = metrics_by_group(y_test, prediction, sensitive)
        rate_gaps = group[["TPR", "FPR"]].max() - group[["TPR", "FPR"]].min()
        gaps[name] = {**rate_gaps.to_dict(), "Equalized odds gap": rate_gaps.max()}
        groups.append(group.assign(Version=name).reset_index())
    pd.DataFrame(performance).T.to_csv(output / "performance_before_after.csv")
    pd.concat(groups).to_csv(output / "metrics_by_group.csv", index=False)
    pd.DataFrame(gaps).T.to_csv(output / "fairness_gaps.csv")
    for name in predictions:
        group = next(g for g in groups if (g.Version == name).all()).set_index("Group")
        group[["TPR", "FPR"]].plot.bar(figsize=(9, 4), ylim=(0, 1))
        plt.title(f"TPR and FPR by race — {name.lower()} mitigation, test")
        plt.ylabel("Rate"); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(output / f"fairness_test_{name.lower()}.png", dpi=160); plt.close()
    summary = {"test_roc_auc": roc_auc_score(y_test, proba),
               "validation_brier": brier_score_loss(data["y_valid"], valid_proba),
               "validation_ece": float(ece), "performance": performance,
               "fairness_gaps": gaps}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
