"""The four baseline notebook models, preprocessing and read-only study lookup."""

import json
import sqlite3
from pathlib import Path

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.distributions import json_to_distribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import NUMERIC_COLUMNS, prepare_data

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_pipeline(columns, name="lightgbm", params=None):
    """columns are the prepared predictor names, in notebook order."""
    categorical = [column for column in columns if column not in NUMERIC_COLUMNS]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    params = dict(params or {})
    if name == "catboost":
        preprocessor = ColumnTransformer([
            ("numeric", numeric["imputer"], NUMERIC_COLUMNS),
            ("categorical", category["imputer"], categorical),
        ], verbose_feature_names_out=False).set_output(transform="pandas")
        model = CatBoostClassifier(
            **({"iterations": 300} | params), cat_features=tuple(categorical),
            random_seed=42, verbose=False, allow_writing_files=False,
        )
    else:
        preprocessor = ColumnTransformer([
            ("numeric", numeric, NUMERIC_COLUMNS), ("categorical", category, categorical),
        ])
        if name == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=42, **params)
        elif name == "forest":
            model = RandomForestClassifier(random_state=42, **params)
        elif name == "lightgbm":
            model = LGBMClassifier(random_state=42, verbosity=-1, **params)
        else:
            raise ValueError("Choose logistic, forest, lightgbm or catboost.")
    return Pipeline([
        ("preparation", FunctionTransformer(prepare_data, validate=False)),
        ("preprocessor", preprocessor), ("model", model),
    ])


def best_trial(database="optuna_results.db", study="lightgbm_tuning"):
    """Extract one-objective maximize study without letting Optuna write to SQLite."""
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT t.trial_id,t.number,v.value FROM trials t "
            "JOIN studies s ON s.study_id=t.study_id "
            "JOIN trial_values v ON v.trial_id=t.trial_id "
            "WHERE s.study_name=? AND t.state='COMPLETE' AND v.objective=0 "
            "ORDER BY v.value DESC,t.number ASC LIMIT 1", (study,),
        ).fetchone()
        if row is None:
            raise ValueError(f"No completed trial for {study} in {database}.")
        parameters = {}
        for name, value, distribution in connection.execute(
            "SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?", (row[0],),
        ):
            parameters[name] = json_to_distribution(distribution).to_external_repr(value)
        attributes = {
            key: json.loads(value) for key, value in connection.execute(
                "SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?", (row[0],),
            )
        }
    return {"study": study, "number": row[1], "value": row[2], "params": parameters, "user_attrs": attributes}
