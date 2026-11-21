"""Create trajectory, parity, and uncertainty-error plots from prediction CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import load_predictions


def make_figure(data, kind, engine=None):
    matplotlib.rcParams.update({"font.family": "serif", "font.size": 10,
                                "axes.spines.top": False, "axes.spines.right": False,
                                "pdf.fonttype": 42, "ps.fonttype": 42})
    if engine is not None:
        if "engine_id" not in data:
            raise ValueError("Engine selection requires an engine_id column.")
        selected = data["engine_id"] == engine
        if not selected.any():
            raise ValueError("Requested engine is absent from the CSV.")
        data = {name: values[selected] for name, values in data.items()}
    figure, axis = plt.subplots(figsize=(5.6, 3.6), layout="constrained")
    if kind == "trajectory":
        if "cycle" not in data:
            raise ValueError("Trajectory plots require a cycle column.")
        if "engine_id" in data and len(np.unique(data["engine_id"])) > 1:
            raise ValueError("Select one engine for a trajectory plot.")
        order = np.argsort(data["cycle"], kind="stable")
        cycle = data["cycle"][order]
        axis.plot(cycle, data["y_true"][order], color="#252525", label="True RUL", linewidth=1.4)
        axis.plot(cycle, data["y_pred"][order], color="#C64B8C", label="Predicted RUL", linewidth=1.3)
        if "lower" in data:
            axis.fill_between(cycle, data["lower"][order], data["upper"][order],
                              color="#E5A6C7", alpha=0.45, label="Prediction interval")
        axis.set(xlabel="Cycle", ylabel="RUL (cycles)")
        axis.legend(frameon=False)
    elif kind == "parity":
        axis.scatter(data["y_true"], data["y_pred"], color="#4477AA", s=14, alpha=0.65)
        low = min(data["y_true"].min(), data["y_pred"].min())
        high = max(data["y_true"].max(), data["y_pred"].max())
        axis.plot([low, high], [low, high], color="#252525", linestyle="--", linewidth=1)
        axis.set(xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)")
    elif kind == "uncertainty-error":
        if "predictive_sd" not in data:
            raise ValueError("Uncertainty plots require predictive_sd.")
        axis.scatter(data["predictive_sd"], np.abs(data["y_pred"] - data["y_true"]),
                     color="#4477AA", s=14, alpha=0.65)
        axis.set(xlabel="Predictive standard deviation", ylabel="Absolute RUL error (cycles)")
    else:
        raise ValueError("Unknown plot kind.")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--kind", choices=["trajectory", "parity", "uncertainty-error"], required=True)
    parser.add_argument("--engine", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() not in (".pdf", ".png", ".svg"):
        parser.error("Output extension must be .pdf, .png, or .svg.")
    data = load_predictions(args.csv)
    figure = make_figure(data, args.kind, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
