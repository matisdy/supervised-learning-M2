"""Export the notebook's framing, cohort summaries and EDA plots from the local snapshot."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_data, split_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--output", default="results/reproduced/eda")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw, y = load_data(args.data)
    cleaned = raw.replace(["?", "None", "Unknown/Invalid"], np.nan)
    counts = raw["patient_nbr"].value_counts()
    parts = split_data(raw, y)
    summary = {
        "source": "Fresh local snapshot summaries; same full cohort as notebook",
        "stays": len(raw), "raw_predictors": len(raw.columns), "patients": int(counts.size),
        "patients_with_multiple_stays": int((counts > 1).sum()),
        "stays_from_repeated_patients": int(counts[counts > 1].sum()),
        "duplicate_rows": int(raw.duplicated().sum()), "positive_fraction": float(y.mean()),
        "target": "Any recorded readmission (<30 or >30), versus NO; predictions at discharge",
        "split_counts": {part: len(parts[f"X_{part}"]) for part in ["development", "fit", "valid", "test"]},
    }
    (output / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({"missing": cleaned.isna().sum(), "missing_percent": cleaned.isna().mean() * 100}).to_csv(output / "missing_values.csv", index_label="feature")
    raw[["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]].describe().to_csv(output / "numeric_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    y.value_counts().sort_index().plot.bar(ax=ax, color=["#6a91b5", "#c97a64"])
    ax.set(title="Target distribution", xlabel="Recorded readmission", ylabel="Number of stays")
    ax.set_xticklabels(["NO", "YES"], rotation=0)
    fig.tight_layout(); fig.savefig(output / "target_distribution.png", dpi=150); plt.close(fig)
    for name, column, title in [
        ("age_distribution", "age", "Age distribution"),
        ("previous_inpatient_distribution", "number_inpatient", "Hospitalizations in the previous year"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=raw, x=column, order=sorted(raw[column].unique()), color="coral", ax=ax)
        ax.set(title=title, ylabel="Number of stays")
        ax.tick_params(axis="x", rotation=45 if column == "age" else 0)
        fig.tight_layout(); fig.savefig(output / f"{name}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=raw, x="time_in_hospital", discrete=True, color="salmon", ax=ax)
    ax.set(title="Length of hospital stays", xlabel="Days in hospital", ylabel="Number of stays")
    fig.tight_layout(); fig.savefig(output / "length_of_stay_distribution.png", dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
