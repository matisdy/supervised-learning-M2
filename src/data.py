"""Load the frozen OpenML snapshot and reproduce notebook partitions."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "previous_visits", "lab_procedures_per_day",
]
DROP_COLUMNS = ["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]


def load_data(path="data/diabetes130us.csv.gz", schema_path=None):
    """Restore OpenML dtypes; CSV parsing must preserve literal None and NO."""
    path = Path(path)
    if schema_path is None:
        schema_path = path.with_name("diabetes130us.json")
    metadata = json.loads(Path(schema_path).read_text())
    expected_hash = metadata.get("snapshot_sha256")
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Dataset checksum differs from the frozen snapshot metadata.")
    data = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if len(data) != metadata["rows"]:
        raise ValueError("Dataset row count differs from the frozen snapshot metadata.")
    target = metadata.get("target", "readmitted")
    if set(data[target].dropna().unique()) != {"NO", "<30", ">30"} or data[target].isna().any():
        raise ValueError("Target must contain exactly NO, <30 and >30, without missing labels.")
    for name, spec in metadata["schema"].items():
        if spec["dtype"] == "category":
            data[name] = pd.Categorical(data[name], categories=spec["categories"],
                                        ordered=spec.get("ordered", False))
        else:
            data[name] = data[name].astype(spec["dtype"])
    target = metadata.get("target", "readmitted")
    y = data.pop(target).map({"NO": 0, "<30": 1, ">30": 1}).astype(int)
    return data, y


def prepare_encounters(data):
    """Apply only fixed, row-wise cleaning and feature calculations."""
    data = pd.DataFrame(data).copy()
    data = data.replace("?", np.nan).replace("None", np.nan)
    data = data.replace("Unknown/Invalid", np.nan)
    data = data.drop(columns=DROP_COLUMNS + ["encounter_id", "patient_nbr"], errors="ignore")
    for name in ["diag_1", "diag_2", "diag_3"]:
        data[name] = data[name].astype("string").fillna("Unknown")
    data["previous_visits"] = (data["number_outpatient"] + data["number_emergency"]
                               + data["number_inpatient"])
    data["lab_procedures_per_day"] = data["num_lab_procedures"] / data["time_in_hospital"]
    return data


class PrepareEncounters(TransformerMixin, BaseEstimator):
    """Run the existing fixed preparation as the first Pipeline step."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return prepare_encounters(X)


def sensitive_groups(data):
    return data["race"].astype("string").replace(["?", "None", "Unknown/Invalid"], pd.NA).fillna("Unknown")


def get_cv():
    return GroupKFold(n_splits=5, shuffle=True, random_state=42)


def split_data(raw, y):
    """Use first fold twice, exactly as notebook cells 53 and 109."""
    development_idx, test_idx = next(get_cv().split(raw, y, groups=raw["patient_nbr"]))
    groups = raw["patient_nbr"].iloc[development_idx]
    X_development = raw.iloc[development_idx].copy()
    X_test = raw.iloc[test_idx].copy()
    y_development, y_test = y.iloc[development_idx], y.iloc[test_idx]
    fit_idx, valid_idx = next(get_cv().split(X_development, y_development, groups=groups))
    assert not set(groups) & set(raw["patient_nbr"].iloc[test_idx])
    assert not set(groups.iloc[fit_idx]) & set(groups.iloc[valid_idx])
    return {
        "X_development": X_development, "y_development": y_development,
        "groups_development": groups, "X_test": X_test, "y_test": y_test,
        "X_fit": X_development.iloc[fit_idx].copy(), "y_fit": y_development.iloc[fit_idx],
        "X_valid": X_development.iloc[valid_idx].copy(), "y_valid": y_development.iloc[valid_idx],
    }
