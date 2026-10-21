"""Descriptive analysis of the full frozen cohort, without model fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import load_data


def run_eda(raw, y, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cleaned = raw.replace("?", np.nan).replace("None", np.nan).replace("Unknown/Invalid", np.nan)
    (cleaned.isna().mean().mul(100).sort_values(ascending=False)
     .rename("Missing percentage").to_csv(output / "missing_percentages.csv"))
    numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
               "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    raw[numeric].describe().to_csv(output / "numeric_summary.csv")
    counts = raw.patient_nbr.value_counts()
    expired = raw.discharge_disposition_id.isin([11, 19, 20, 21])
    summary = {"rows": len(raw), "predictors": raw.shape[1], "unique_patients": len(counts),
               "repeated_patients": int((counts > 1).sum()),
               "stays_of_repeated_patients": int(counts[counts > 1].sum()),
               "duplicated_predictor_rows": int(raw.duplicated().sum()),
               "positive_rate": float(y.mean()), "expired_stays": int(expired.sum()),
               "positive_expired_stays": int(y[expired].sum())}
    (output / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    y.value_counts().sort_index().plot.bar()
    plt.xticks([0, 1], ["NO", "YES"], rotation=0); plt.ylabel("Number of stays")
    plt.title("Distribution of recorded readmission"); plt.tight_layout()
    plt.savefig(output / "target_distribution.png", dpi=160); plt.close()
    for column, title in [("age", "Age distribution"), ("time_in_hospital", "Length of hospital stays"),
                          ("number_inpatient", "Hospitalisations in the previous year")]:
        raw[column].value_counts().sort_index().plot.bar(figsize=(8, 4))
        plt.title(title); plt.ylabel("Number of stays"); plt.xlabel(column)
        plt.tight_layout(); plt.savefig(output / f"{column}_distribution.png", dpi=160); plt.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--output", default="results/reproduced/eda")
    args = parser.parse_args()
    raw, y = load_data(args.data)
    print(json.dumps(run_eda(raw, y, args.output), indent=2))

if __name__ == "__main__":
    main()
