"""Load the frozen OpenML dataset and reproduce notebook cells 13, 30–35, 53–69, 111."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "previous_visits", "lab_procedures_per_day",
]
DROP_COLUMNS = [
    "weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code",
    "encounter_id", "patient_nbr",
]
TARGET_MAPPING = {"NO": 0, "<30": 1, ">30": 1}


def restore_schema(data, schema, preserve_unknown=False):
    """Restore snapshot types; object categories can retain unseen endpoint values."""
    data = data.copy()
    for column in data.columns:
        spec = schema[column]
        if spec["dtype"] == "category":
            if preserve_unknown:
                data[column] = data[column].astype(object)
            else:
                data[column] = pd.Categorical(
                    data[column], categories=spec["categories"],
                    ordered=spec.get("ordered", False),
                )
        else:
            data[column] = pd.to_numeric(data[column], errors="raise").astype(spec["dtype"])
    return data


def load_data(path="data/diabetes130us.csv.gz"):
    """Read local data only; preserve literal NO/None/? and diagnosis strings."""
    path = Path(path)
    info = json.loads(path.with_suffix("").with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["snapshot_sha256"]:
        raise ValueError("Snapshot SHA-256 differs from the supplied dataset metadata.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = restore_schema(frame, info["schema"])
    target = frame.pop(info["target"]).map(TARGET_MAPPING).astype(int)
    if len(frame) != info["rows"] or target.isna().any():
        raise ValueError("Dataset does not match the supplied OpenML snapshot metadata.")
    return frame, target


def prepare_data(data):
    """Notebook cell 88: deterministic cleaning and two engineered features."""
    data = data.copy()
    data = data.replace("?", np.nan)
    data = data.replace("None", np.nan)
    data = data.replace("Unknown/Invalid", np.nan)
    data = data.drop(columns=DROP_COLUMNS, errors="ignore")
    for column in ["diag_1", "diag_2", "diag_3"]:
        data[column] = data[column].astype("string").fillna("Unknown")
    data["previous_visits"] = (
        data["number_outpatient"] + data["number_emergency"] + data["number_inpatient"]
    )
    data["lab_procedures_per_day"] = data["num_lab_procedures"] / data["time_in_hospital"]
    return data


def sensitive_race(data):
    """Use the notebook's cleaned race categories, including Unknown."""
    return (
        data["race"].astype("string")
        .replace(["?", "None", "Unknown/Invalid"], pd.NA).fillna("Unknown")
    )


def grouped_cv():
    return GroupKFold(n_splits=5, shuffle=True, random_state=42)


def split_data(raw, y):
    """Keep the existing protocol: tuning uses all development, then fit/valid split."""
    development, test = next(grouped_cv().split(raw, y, groups=raw["patient_nbr"]))
    X_development, y_development = raw.iloc[development].copy(), y.iloc[development].copy()
    groups_development = X_development["patient_nbr"].copy()
    fit, valid = next(grouped_cv().split(X_development, y_development, groups=groups_development))
    result = {
        "X_development": X_development, "y_development": y_development,
        "groups_development": groups_development,
        "X_fit": X_development.iloc[fit].copy(), "y_fit": y_development.iloc[fit].copy(),
        "X_valid": X_development.iloc[valid].copy(), "y_valid": y_development.iloc[valid].copy(),
        "X_test": raw.iloc[test].copy(), "y_test": y.iloc[test].copy(),
    }
    patient_sets = [set(result[f"X_{part}"]["patient_nbr"]) for part in ["fit", "valid", "test"]]
    assert not (patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2])
    return result
