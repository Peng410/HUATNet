"""Evaluate RUL predictions supplied as a CSV file."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_predictions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not {"y_true", "y_pred"}.issubset(columns):
            raise ValueError("CSV requires unique headers including y_true and y_pred.")
        rows = list(reader)
    if not rows:
        raise ValueError("Prediction CSV is empty.")
    data = {}
    for name in ("y_true", "y_pred", "lower", "upper", "predictive_sd", "cycle", "engine_id"):
        if name in columns:
            data[name] = np.array([float(row[name]) for row in rows], dtype=np.float64)
            if not np.isfinite(data[name]).all():
                raise ValueError(f"Column {name} must contain finite values.")
    if ("lower" in data) != ("upper" in data):
        raise ValueError("Provide both lower and upper interval bounds.")
    if "lower" in data and (data["upper"] < data["lower"]).any():
        raise ValueError("Interval upper bounds must be at least the lower bounds.")
    if "predictive_sd" in data and (data["predictive_sd"] < 0).any():
        raise ValueError("Predictive standard deviation must be nonnegative.")
    return data


def prediction_metrics(data):
    error = data["y_pred"] - data["y_true"]
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    with np.errstate(over="ignore"):
        score = float(np.expm1(exponent).sum())
    if not np.isfinite(score):
        raise ValueError("NASA Score overflow; check prediction units and magnitudes.")
    result = {"samples": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))),
              "mae": float(np.abs(error).mean()), "nasa_score": score}
    if "lower" in data:
        result["picp"] = float(((data["y_true"] >= data["lower"]) & (data["y_true"] <= data["upper"])).mean())
        result["mpiw"] = float((data["upper"] - data["lower"]).mean())
    if "predictive_sd" in data:
        uncertainty, absolute_error = data["predictive_sd"], np.abs(error)
        result["pearson_sd_absolute_error"] = None
        if len(error) > 1 and np.std(uncertainty) > 0 and np.std(absolute_error) > 0:
            result["pearson_sd_absolute_error"] = float(np.corrcoef(uncertainty, absolute_error)[0, 1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(prediction_metrics(load_predictions(args.csv)), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
