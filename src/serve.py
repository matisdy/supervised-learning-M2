"""Local proof-of-concept HTTP transport for the frozen notebook model.

Run from the project root: python -m src.serve --model artifacts/model.joblib
Only trusted, locally generated joblib bundles should be loaded.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import sensitive_groups


def validate_request(payload, metadata):
    """Validate JSON against the frozen raw schema without inventing categories."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    extra = set(payload) - {"records", "mitigation", "random_seed"}
    if extra:
        raise ValueError("Unknown request fields: " + ", ".join(sorted(extra)))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty array of encounter objects.")
    mitigation = payload.get("mitigation", False)
    if not isinstance(mitigation, bool):
        raise ValueError("mitigation must be a boolean.")
    seed = payload.get("random_seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("random_seed must be null or an integer between 0 and 4294967295.")
    columns = metadata["raw_columns"]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object.")
        missing, unexpected = set(columns) - set(record), set(record) - set(columns)
        if missing:
            raise ValueError(f"records[{index}] missing columns: " + ", ".join(sorted(missing)))
        if unexpected:
            raise ValueError(f"records[{index}] unexpected columns: " + ", ".join(sorted(unexpected)))
        for column in columns:
            value = record[column]
            dtype = metadata["raw_schema"][column]["dtype"]
            if dtype == "category":
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"records[{index}].{column} must be a string or null.")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"records[{index}].{column} must be a finite number.")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(f"records[{index}].{column} must be a finite number.")
                if dtype.startswith("int") and (int(value) != value or not -(2**63) <= value < 2**63):
                    raise ValueError(f"records[{index}].{column} must be a signed 64-bit integer value.")
        if record["time_in_hospital"] <= 0:
            raise ValueError(f"records[{index}].time_in_hospital must be greater than zero.")
    raw = pd.DataFrame(records, columns=columns).replace({None: np.nan})
    for column in columns:
        dtype = metadata["raw_schema"][column]["dtype"]
        if dtype != "category":
            raw[column] = raw[column].astype(dtype)
    if mitigation:
        groups = sensitive_groups(raw)
        unknown = set(groups) - set(metadata["sensitive_groups"])
        if unknown:
            raise ValueError("Mitigation requires a race group seen during fitting; unknown groups: "
                             + ", ".join(sorted(unknown)))
    return raw, mitigation, seed


def predict(payload, bundle):
    raw, mitigation, seed = validate_request(payload, bundle["metadata"])
    probabilities = bundle["pipeline"].predict_proba(raw)[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    predictions = [{"probability": float(p), "prediction": int(label)}
                   for p, label in zip(probabilities, labels)]
    if mitigation:
        groups = sensitive_groups(raw)
        decisions = bundle["fair_model"].predict(
            raw, sensitive_features=groups, random_state=seed
        )
        for result, decision in zip(predictions, decisions):
            result["mitigated_prediction"] = int(decision)
    return {"predictions": predictions, "threshold": 0.5,
            "mitigation": mitigation,
            "target": bundle["metadata"]["target"]}


def make_handler(bundle):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            content = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "Unknown route."})
            status = 200 if bundle is not None else 503
            self.send_json(status, {"status": "ready" if bundle is not None else "unavailable",
                                    "model_loaded": bundle is not None})

        def do_POST(self):
            if self.path != "/predict":
                return self.send_json(404, {"error": "Unknown route."})
            if bundle is None:
                return self.send_json(503, {"error": "Model unavailable. Rebuild artifacts/model.joblib."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("A non-empty JSON body with Content-Length is required.")
                body = self.rfile.read(length).decode("utf-8")
                def reject_constant(value):
                    raise ValueError("NaN and Infinity are not valid JSON numbers.")
                payload = json.loads(body, parse_constant=reject_constant)
                result = predict(payload, bundle)
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                return self.send_json(400, {"error": str(error)})
            except Exception:
                return self.send_json(503, {"error": "Prediction unavailable; inspect the model and environment locally."})
            self.send_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="artifacts/model.joblib", type=Path)
    args = parser.parse_args()
    try:
        bundle = joblib.load(args.model)
        for key in ("pipeline", "fair_model", "metadata"):
            if key not in bundle:
                raise ValueError("Invalid model bundle.")
    except Exception:
        bundle = None
        print("Model unavailable. Rebuild with python -m src.train.", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bundle))
    print(f"Local proof of concept listening on http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
