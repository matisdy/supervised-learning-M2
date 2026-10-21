"""Reproduce gain importance and SHAP global/local explanations from notebook section 8."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import load_data, split_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/diabetes130us.csv.gz")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output", default="results/reproduced/explainability")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    model = pipeline.named_steps["model"]
    raw, y = load_data(args.data)
    parts = split_data(raw, y)
    sample = parts["X_development"].sample(300, random_state=42)
    transformed = pipeline[:-1].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    transformed = pd.DataFrame(transformed, columns=feature_names, index=sample.index)
    importance = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feature_names, name="gain")
    importance.sort_values(ascending=False).to_csv(output / "feature_importance_gain.csv", index_label="feature")
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.nlargest(10).sort_values().plot.barh(ax=ax)
    ax.set(title="Top 10 LightGBM feature importances", xlabel="Total gain")
    fig.tight_layout(); fig.savefig(output / "feature_importance_top10.png", dpi=150); plt.close(fig)
    explainer = shap.TreeExplainer(model, model_output="raw")
    values = explainer.shap_values(transformed)
    shap.summary_plot(values, transformed, max_display=10, show=False)
    plt.tight_layout(); plt.savefig(output / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    pd.Series(np.abs(values).mean(axis=0), index=feature_names, name="mean_absolute_shap").sort_values(ascending=False).to_csv(output / "shap_global.csv", index_label="feature")
    np.savez_compressed(output / "shap_values.npz", values=values, row_indices=sample.index.to_numpy(), feature_names=feature_names.astype(str))
    local = []
    for position in [0, 1, 2]:
        explanation = pd.DataFrame({"Feature": feature_names, "Value": transformed.iloc[position].values, "SHAP": values[position]})
        explanation["Absolute SHAP"] = explanation["SHAP"].abs()
        top = explanation.nlargest(5, "Absolute SHAP")
        top.to_csv(output / f"local_stay_{sample.index[position]}.csv", index=False)
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        reconstructed = baseline + float(values[position].sum())
        raw_score = float(model.predict(transformed.iloc[[position]], raw_score=True)[0])
        check = bool(np.isclose(reconstructed, raw_score))
        if not check:
            raise AssertionError("SHAP additivity failed.")
        local.append({
            "row_index": int(sample.index[position]), "probability": float(pipeline.predict_proba(sample.iloc[[position]])[0, 1]),
            "baseline_raw_score": baseline, "reconstructed_raw_score": reconstructed,
            "model_raw_score": raw_score, "additivity_check": check,
        })
    (output / "local_checks.json").write_text(json.dumps(local, indent=2) + "\n")
    print(json.dumps(local, indent=2))


if __name__ == "__main__":
    main()
