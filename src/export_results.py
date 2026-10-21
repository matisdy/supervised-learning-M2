"""Export recorded notebook outputs and Optuna evidence without training.

SQLite is opened in read-only mode. Saved outputs retain their original or explicitly labelled reproduction provenance;
this command does not execute the notebook. Run from the repository root with
``python -m src.export_results``.
"""

import argparse
import ast
import base64
import hashlib
import json
import math
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FOLD_KEYS = ["fit_time", "score_time", "test_accuracy", "test_precision",
             "test_recall", "test_f1", "test_roc_auc"]
TABLE_CELLS = {84: "baseline_comparison_rounded", 102: "best_trials_folds_rounded",
               121: "local_shap", 135: "validation_fairness_rounded",
               144: "test_performance_rounded", 146: "test_fairness_by_group_rounded",
               148: "test_fairness_gaps_rounded"}
FIGURE_LABELS = {27: "Missingness before cleaning", 36: "Missingness after cleaning",
                 42: "Binary target distribution", 44: "Age distribution",
                 46: "Length of hospital stays", 48: "Previous hospitalizations",
                 108: "Optuna optimization history (LightGBM, then CatBoost)",
                 117: "LightGBM top-ten gain importances", 119: "SHAP summary",
                 126: "Validation calibration curve", 129: "Validation probability distribution",
                 138: "Validation TPR/FPR by race before mitigation"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_value(value):
    return "".join(value) if isinstance(value, list) else value


class DisplayTable(HTMLParser):
    """Read the displayed DataFrame grid, including repeated index labels."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.header = []
        self.in_header = False
        self.active = None
        self.pending = {}

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.in_header = True
        elif tag == "tr":
            self.row = []
            self.column = 0
        elif tag in ("th", "td"):
            while self.column in self.pending:
                value, remaining = self.pending.pop(self.column)
                self.row.append(value)
                if remaining > 1:
                    self.pending[self.column] = (value, remaining - 1)
                self.column += 1
            self.active = []
            self.span = int(dict(attrs).get("rowspan", 1))

    def handle_data(self, data):
        if self.active is not None:
            self.active.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.active is not None:
            value = "".join(self.active).strip()
            self.row.append(value)
            if self.span > 1:
                self.pending[self.column] = (value, self.span - 1)
            self.column += 1
            self.active = None
        elif tag == "tr":
            (self.header if self.in_header else self.rows).append(self.row)
        elif tag == "thead":
            self.in_header = False

    def dataframe(self):
        names = [next((row[i] for row in self.header if row[i]), f"index_{i}")
                 for i in range(len(self.header[0]))]
        return pd.DataFrame(self.rows, columns=names)


def recorded_text(cell):
    return "\n".join(text_value(out.get("text", out.get("data", {}).get("text/plain", "")))
                     for out in cell.get("outputs", []))


def export_notebook(path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(path.read_text())
    manifest = {"evidence_type": "saved_notebook_outputs_with_labelled_reproduction",
                "source": path.name, "sha256": sha256(path), "cell_count": len(notebook["cells"]),
                "cell_numbers": "1-based, including Markdown", "files": []}
    errors = []
    for cell_no, cell in enumerate(notebook["cells"], 1):
        table_no = 0
        for output_no, output in enumerate(cell.get("outputs", []), 1):
            if output.get("output_type") == "error":
                errors.append({"cell": cell_no, "ename": output.get("ename")})
            data = output.get("data", {})
            if "image/png" in data:
                filename = f"cell_{cell_no:03d}_output_{output_no:02d}.png"
                image_bytes = base64.b64decode(text_value(data["image/png"]), validate=True)
                if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"Invalid PNG at cell {cell_no}")
                (output_dir / filename).write_bytes(image_bytes)
                manifest["files"].append({"file": filename, "cell": cell_no,
                                          "output": output_no, "type": "unchanged_embedded_png",
                                          "description": FIGURE_LABELS.get(cell_no, "Recorded notebook figure")})
            if cell_no in TABLE_CELLS and "text/html" in data:
                table_no += 1
                parser = DisplayTable()
                parser.feed(text_value(data["text/html"]))
                filename = f"{TABLE_CELLS[cell_no]}_{table_no}.csv"
                frame = parser.dataframe()
                frame.to_csv(output_dir / filename, index=False)
                manifest["files"].append({"file": filename, "cell": cell_no,
                                          "output": output_no, "type": "displayed_table",
                                          "rows": len(frame), "precision": "as displayed; not full precision"})

    baseline_rows = []
    for model, cell_no in [("Logistic Regression", 80), ("Random Forest", 81), ("LightGBM", 82), ("CatBoost", 83)]:
        cell = notebook["cells"][cell_no - 1]
        plain = "\n".join(text_value(out.get("data", {}).get("text/plain", ""))
                          for out in cell.get("outputs", []))
        values = dict((key, float(value)) for key, value in re.findall(
            r"^(fit_time|score_time|test_\w+)\s+([0-9.eE+-]+)\s*$", plain, re.M))
        if len(values) != 7:
            raise ValueError(f"Missing baseline means at cell {cell_no}")
        baseline_rows.append({"model": model, "source_cell": cell_no, "provenance": cell.get("metadata", {}).get("delivery_output_provenance", "original_saved_output"), **values})
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_cv_means_displayed.csv", index=False)
    manifest["files"].append({"file": "baseline_cv_means_displayed.csv", "cells": [80, 81, 82, 83],
                              "precision": "six displayed decimals; restored RF fold results are in results/reproduced/comparison_folds.csv"})

    # Extract constructor arguments from code, without executing any model.
    parameters = {}
    tree = ast.parse(text_value(notebook["cells"][75]["source"]))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            settings = {}
            for keyword in node.value.keywords:
                try:
                    settings[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    settings[keyword.arg] = {"expression": ast.unparse(keyword.value)}
            name = node.targets[0].id
            parameters[name] = {"constructor": ast.unparse(node.value.func), "arguments": settings,
                                "source_cell": 76, "included_in_comparison": True}
    (output_dir / "baseline_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    manifest["files"].append({"file": "baseline_parameters.json", "cell": 76,
                              "type": "declared_constructor_arguments_not_full_library_defaults"})

    source_cells = [96, 100, 114, 121, 122, 126, 127, 137, 144]
    evidence = {str(i): {"execution_count": notebook["cells"][i - 1].get("execution_count"),
                         "text": recorded_text(notebook["cells"][i - 1])}
                for i in source_cells}
    (output_dir / "recorded_text.json").write_text(json.dumps(evidence, indent=2) + "\n")
    manifest["files"].append({"file": "recorded_text.json", "cells": source_cells,
                              "type": "unchanged_saved_text"})
    def scalar(cell, label):
        match = re.search(re.escape(label) + r"\s*([0-9.eE+-]+)", evidence[str(cell)]["text"])
        if not match:
            raise ValueError(f"Missing recorded metric {label} at cell {cell}")
        return {"value": float(match.group(1)), "source_cell": cell,
                "precision": "saved float representation"}

    def rounded_table(filename, cell):
        table = pd.read_csv(output_dir / filename, index_col=0)
        return {"values": table.T.to_dict(), "source_cell": cell,
                "decimal_places": 3, "rounding_tolerance": 0.0005}

    metrics = {"evidence_type": "saved_notebook_outputs_with_labelled_reproduction", "source_sha256": sha256(path),
               "validation_calibration": {"brier": scalar(126, "Brier score:"), "ece": scalar(127, "ECE:")},
               "test_roc_auc": scalar(144, "Test ROC-AUC:"),
               "test_performance": rounded_table("test_performance_rounded_1.csv", 144),
               "test_fairness_gaps": rounded_table("test_fairness_gaps_rounded_1.csv", 148),
               "baseline_cv_means": {"values": baseline_rows, "decimal_places": 6,
                                     "rounding_tolerance": 0.0000005},
               "active_baselines": [row["model"] for row in baseline_rows]}
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest["files"].append({"file": "metrics_summary.json", "cells": [80, 81, 82, 83, 126, 127, 144, 148],
                              "type": "parsed_recorded_metrics_with_precision"})
    manifest["error_outputs"] = errors
    manifest["figure_count"] = sum(item.get("type") == "unchanged_embedded_png" for item in manifest["files"])
    manifest["table_count"] = sum(item.get("type") == "displayed_table" for item in manifest["files"])
    manifest["notes"] = ["Random Forest cell 81 and comparison cell 84 contain explicitly labelled script reproduction outputs; other saved outputs remain historical.",
                         "Execution counters and saved outputs do not prove a complete fresh top-to-bottom run."]
    for item in manifest["files"]:
        item["sha256"] = sha256(output_dir / item["file"])
    return manifest


def parameter_value(raw, distribution):
    spec = json.loads(distribution)
    if spec["name"] == "CategoricalDistribution":
        return spec["attributes"]["choices"][int(raw)]
    return int(raw) if spec["name"] == "IntDistribution" else float(raw)


def export_optuna(database, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(database)
    summary = {"evidence_type": "historical_sqlite_trials", "source": database.name,
               "sha256": source_hash, "timestamp_note": "Raw database timestamps, timezone unspecified. No conversion applied.",
               "studies": []}
    # Read-only SQLite avoids Optuna schema checks that could write the database.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for study in connection.execute("SELECT study_id, study_name FROM studies ORDER BY study_id").fetchall():
            study_id, study_name = study["study_id"], study["study_name"]
            trials, fold_rows = [], []
            for trial in connection.execute("SELECT * FROM trials WHERE study_id=? ORDER BY number", (study_id,)).fetchall():
                trial_id = trial["trial_id"]
                value = connection.execute("SELECT value FROM trial_values WHERE trial_id=? AND objective=0", (trial_id,)).fetchone()
                row = {key: trial[key] for key in ["number", "state", "datetime_start", "datetime_complete"]}
                row["value"] = value[0] if value else None
                params = {}
                for param in connection.execute("SELECT * FROM trial_params WHERE trial_id=?", (trial_id,)):
                    params[param["param_name"]] = parameter_value(param["param_value"], param["distribution_json"])
                row.update({"params_" + key: val for key, val in params.items()})
                attrs = {attr["key"]: json.loads(attr["value_json"]) for attr in connection.execute(
                    "SELECT * FROM trial_user_attributes WHERE trial_id=?", (trial_id,))}
                row.update({"user_attrs_" + key: json.dumps(val) for key, val in attrs.items()})
                for key in FOLD_KEYS:
                    if len(attrs.get(key, [])) != 5:
                        raise ValueError(f"{study_name} trial {row['number']}: expected five {key} values")
                if not math.isclose(sum(attrs["test_roc_auc"]) / 5, row["value"], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Objective/fold disagreement in {study_name} trial {row['number']}")
                for fold in range(5):
                    fold_rows.append({"trial_number": row["number"], "fold": fold + 1,
                                      **{key: attrs[key][fold] for key in FOLD_KEYS}})
                trials.append(row)
            frame = pd.DataFrame(trials)
            frame.to_csv(output_dir / f"{study_name}_trials.csv", index=False)
            pd.DataFrame(fold_rows).to_csv(output_dir / f"{study_name}_folds.csv", index=False)
            complete = frame[frame["state"] == "COMPLETE"]
            best = complete.loc[complete["value"].idxmax()]
            best_original = next(row for row in trials if row["number"] == best["number"])
            summary["studies"].append({"study_name": study_name, "trial_count": len(frame),
                                       "state_counts": frame["state"].value_counts().to_dict(),
                                       "fold_rows": len(fold_rows), "best_trial": int(best["number"]),
                                       "best_roc_auc": float(best["value"]),
                                       "best_params": {key.removeprefix("params_"): val
                                                       for key, val in best_original.items()
                                                       if key.startswith("params_")}})
    if sha256(database) != source_hash:
        raise RuntimeError("Source database changed during export")
    summary["source_unchanged_after_export"] = True
    summary["files"] = [{"file": file.name, "sha256": sha256(file)}
                        for file in sorted(output_dir.glob("*.csv"))]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "diabetes.ipynb")
    parser.add_argument("--database", type=Path, default=ROOT / "optuna_results.db")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    notebook = export_notebook(args.notebook, args.output / "notebook")
    studies = export_optuna(args.database, args.output / "optuna")
    print(f"Exported {notebook['figure_count']} recorded figures and {notebook['table_count']} displayed tables.")
    print(f"Exported {sum(study['trial_count'] for study in studies['studies'])} historical Optuna trials; database unchanged.")


if __name__ == "__main__":
    main()
