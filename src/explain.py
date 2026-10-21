"""Reproduce the notebook's 300-stay SHAP sample and local checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def explain(pipeline, X_development, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    preprocessor, model = pipeline["preprocessor"], pipeline["model"]
    names = preprocessor.get_feature_names_out()
    importance = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=names)
    importance.sort_values(ascending=False).to_csv(output / "feature_importance_gain.csv")
    importance.nlargest(10).sort_values().plot.barh(figsize=(8, 5))
    plt.xlabel("Total gain"); plt.title("Top 10 LightGBM feature importances"); plt.tight_layout()
    plt.savefig(output / "feature_importance.png", dpi=160); plt.close()
    sample = X_development.sample(300, random_state=42)
    transformed = pipeline[:-1].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = pd.DataFrame(transformed, columns=names, index=sample.index)
    explainer = shap.TreeExplainer(model, model_output="raw")
    values = explainer.shap_values(transformed)
    shap.summary_plot(values, transformed, max_display=10, show=False)
    plt.tight_layout(); plt.savefig(output / "shap_summary.png", dpi=160); plt.close()
    local = []
    for position in [0, 1, 2]:
        table = pd.DataFrame({"Feature": names, "Value": transformed.iloc[position].values,
                              "SHAP": values[position]})
        table["Absolute SHAP"] = table.SHAP.abs()
        table.sort_values("Absolute SHAP", ascending=False).head(5).to_csv(
            output / f"local_shap_{sample.index[position]}.csv", index=False)
        reconstructed = np.asarray(explainer.expected_value + values[position].sum()).item()
        raw_score = model.predict(transformed.iloc[[position]], raw_score=True)[0]
        local.append({"stay_index": int(sample.index[position]),
                      "probability": float(pipeline.predict_proba(sample.iloc[[position]])[0, 1]),
                      "additivity_check": bool(np.isclose(reconstructed, raw_score))})
    (output / "local_explanations.json").write_text(json.dumps(local, indent=2))
    assert all(item["additivity_check"] for item in local)
    return local
