"""Serve the frozen notebook pipeline locally with Python's HTTP server."""

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd


DROPPED_COLUMNS = {
    "weight", "max_glu_serum", "A1Cresult", "medical_specialty",
    "payer_code", "encounter_id", "patient_nbr",
}


class InputError(ValueError):
    """A request cannot be used by the frozen model."""


def load_bundle(path):
    """Load a locally generated artifact, without training or downloading."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run python -m src.train first."
        )
    bundle = joblib.load(path)
    for key in ["pipeline", "input_columns", "schema", "metadata"]:
        if key not in bundle:
            raise ValueError(f"Model artifact is missing {key!r}; recreate it with src.train.")
    return bundle


def request_frame(records, bundle):
    """Validate JSON types and preserve OpenML numeric/category conventions."""
    if not isinstance(records, list) or not records:
        raise InputError("records must be a non-empty list of objects.")
    if len(records) > 1000:
        raise InputError("At most 1000 records are accepted per request.")
    columns = bundle["input_columns"]
    required = set(columns) - DROPPED_COLUMNS
    allowed = set(columns)
    schema = bundle["schema"]
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise InputError(f"Record {index} must be an object.")
        missing = required - set(record)
        if missing:
            raise InputError(f"Record {index} is missing columns: {', '.join(sorted(missing))}.")
        extra = set(record) - allowed
        if extra:
            raise InputError(
                f"Record {index} has unexpected columns: {', '.join(sorted(extra))}. "
                "Do not include readmitted or engineered features."
            )
        row = {}
        for name in columns:
            value = record.get(name)
            if name in DROPPED_COLUMNS:
                # These columns are discarded by the pipeline and may be omitted.
                row[name] = value
                continue
            dtype = schema[name]["dtype"]
            if dtype.startswith(("int", "float")):
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise InputError(f"Record {index}: {name} must be a finite number or null.")
                if value is not None and dtype.startswith("int") and value != int(value):
                    raise InputError(f"Record {index}: {name} must be an integer or null.")
                row[name] = np.nan if value is None else value
            else:
                if value is not None and not isinstance(value, str):
                    raise InputError(f"Record {index}: {name} must be a string or null.")
                # Object dtype preserves unseen string categories for handle_unknown.
                row[name] = np.nan if value is None else value
        duration = row["time_in_hospital"]
        if pd.isna(duration) or duration <= 0:
            raise InputError(f"Record {index}: time_in_hospital must be greater than zero.")
        normalized.append(row)
    frame = pd.DataFrame(normalized, columns=columns)
    for name in columns:
        dtype = schema[name]["dtype"]
        if name in DROPPED_COLUMNS or not dtype.startswith(("int", "float")):
            frame[name] = frame[name].astype(object)
        elif frame[name].notna().all():
            frame[name] = frame[name].astype(dtype)
    return frame


def predict(payload, bundle):
    if not isinstance(payload, dict) or set(payload) - {"records", "mitigated"}:
        raise InputError("Body must be an object containing records and optional mitigated.")
    mitigated = payload.get("mitigated", False)
    if not isinstance(mitigated, bool):
        raise InputError("mitigated must be true or false.")
    frame = request_frame(payload.get("records"), bundle)
    probabilities = np.asarray(bundle["pipeline"].predict_proba(frame))[:, 1]
    if mitigated:
        if bundle.get("mitigator") is None:
            raise InputError("This artifact does not contain a fitted mitigation model.")
        sensitive = (
            frame["race"].astype("string")
            .replace(["?", "None", "Unknown/Invalid"], pd.NA)
            .fillna("Unknown")
        )
        known = set(bundle["mitigator"].interpolated_thresholder_.interpolation_dict)
        unknown = set(sensitive) - known
        if unknown:
            raise InputError("Mitigation has no fitted rule for race: " + ", ".join(sorted(unknown)))
        predictions = bundle["mitigator"].predict(
            frame, sensitive_features=sensitive, random_state=42
        )
    else:
        predictions = (probabilities >= 0.5).astype(int)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced a non-finite probability.")
    return {
        "probabilities": probabilities.tolist(),
        "predictions": np.asarray(predictions, dtype=int).tolist(),
        "mitigated": mitigated,
    }


def make_server(bundle, host="127.0.0.1", port=8000):
    """Construct the local server; used by the CLI and the HTTP integration check."""
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit(self.path).path != "/health":
                self.respond(404, {"error": "Unknown route. Use GET /health or POST /predict."})
                return
            self.respond(200, {"status": "ok", "model_loaded": True})

        def do_POST(self):
            if urlsplit(self.path).path != "/predict":
                self.respond(404, {"error": "Unknown route. Use GET /health or POST /predict."})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "Content-Type must be application/json."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 2_000_000:
                    raise InputError("JSON body must contain 1 to 2000000 bytes.")
                payload = json.loads(
                    self.rfile.read(size),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        InputError(f"Invalid JSON numeric constant: {value}.")
                    ),
                )
                result = predict(payload, bundle)
            except (InputError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.respond(400, {"error": f"Input cannot be processed: {exc}"})
                return
            except Exception:
                self.respond(500, {"error": "Prediction failed; inspect the local server log."})
                raise
            self.respond(200, result)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        bundle = load_bundle(args.model)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    with make_server(bundle, args.host, args.port) as server:
        print(f"Model loaded. Listening on http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
