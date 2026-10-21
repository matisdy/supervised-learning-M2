"""Course-style preprocessing Pipelines and candidate classifiers."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .data import NUMERIC_COLUMNS, DROP_COLUMNS, PrepareEncounters

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_preprocessor(columns, catboost=False):
    categorical = [name for name in columns if name not in NUMERIC_COLUMNS]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    category = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    if catboost:
        return ColumnTransformer([
            ("numeric", numeric["imputer"], NUMERIC_COLUMNS),
            ("categorical", category["imputer"], categorical)
        ], verbose_feature_names_out=False).set_output(transform="pandas")
    return ColumnTransformer([("numeric", numeric, NUMERIC_COLUMNS),
                              ("categorical", category, categorical)])


def make_pipeline(columns, name="lightgbm", params=None):
    columns = [c for c in columns if c not in DROP_COLUMNS + ["encounter_id", "patient_nbr"]]
    columns += [c for c in ["previous_visits", "lab_procedures_per_day"] if c not in columns]
    params = {} if params is None else params.copy()
    categorical = [name for name in columns if name not in NUMERIC_COLUMNS]
    if name == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42, **params)
    elif name == "forest":
        model = RandomForestClassifier(random_state=42, **params)
    elif name == "lightgbm":
        model = LGBMClassifier(random_state=42, verbosity=-1, **params)
    elif name == "catboost":
        params.setdefault("iterations", 300)
        model = CatBoostClassifier(cat_features=tuple(categorical), random_seed=42,
                                  verbose=False, allow_writing_files=False, **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preparation", PrepareEncounters()),
                     ("preprocessor", make_preprocessor(columns, name == "catboost")),
                     ("model", model)])
