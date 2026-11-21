"""Prepare C-MAPSS windows using training-fitted condition normalization."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_trajectories(path):
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.shape[1] != 26 or not np.isfinite(values).all():
        raise ValueError("Expected 26 finite columns in a C-MAPSS trajectory file.")
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    if (values[:, :2] < 1).any() or not np.equal(values[:, :2], np.floor(values[:, :2])).all():
        raise ValueError("Engine IDs and cycles must be positive integers.")
    for unit in np.unique(values[:, 0]):
        cycles = values[values[:, 0] == unit, 1]
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            raise ValueError("Each engine must have consecutive cycles starting at one.")
    return values


def prefix_window(values, end, length):
    """Read through the requested endpoint and repeat the first row on the left."""
    if length < 1 or not 1 <= end <= len(values):
        raise ValueError("Invalid window length or endpoint.")
    window = values[max(0, end - length):end]
    padding = length - len(window)
    if padding:
        window = np.concatenate([np.repeat(window[:1], padding, axis=0), window])
    return window


def scaler_state(scaler):
    return {"scale": scaler.scale_.tolist(), "offset": scaler.min_.tolist(),
            "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist()}


def prepare_windows(train, test, final_rul, config):
    length, cap = config["window_length"], config["max_rul"]
    sensor_ids = config["sensor_ids"]
    if not isinstance(length, int) or length < 1 or not np.isfinite(cap) or cap <= 0:
        raise ValueError("Window length and maximum RUL must be positive.")
    if not sensor_ids or len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Sensor IDs must be nonempty and unique.")
    if any(not isinstance(value, int) or not 1 <= value <= 21 for value in sensor_ids):
        raise ValueError("Sensor IDs must be integers from 1 through 21.")
    columns = [4 + sensor for sensor in sensor_ids]
    decimals = config["condition_decimals"]
    groups = np.round(train[:, 2], decimals)
    scalers = {float(group): MinMaxScaler().fit(train[groups == group][:, columns])
               for group in np.unique(groups)}
    operating_scaler = MinMaxScaler().fit(train[:, 2:5])
    test_units = np.unique(test[:, 0]).astype(int)
    final_rul = np.asarray(final_rul, dtype=np.float64).reshape(-1)
    if len(final_rul) != len(test_units) or not np.isfinite(final_rul).all() or (final_rul < 0).any():
        raise ValueError("The RUL file must contain one nonnegative label per test engine.")
    truth = dict(zip(test_units, final_rul))
    prepared = {}
    for split, raw in (("train", train), ("test", test)):
        normalized = np.empty((len(raw), len(columns)), dtype=np.float32)
        split_groups = np.round(raw[:, 2], decimals)
        for group in np.unique(split_groups):
            if float(group) not in scalers:
                raise ValueError(f"Operating group {group} is absent from the training data.")
            selected = split_groups == group
            normalized[selected] = scalers[float(group)].transform(raw[selected][:, columns])
        settings = operating_scaler.transform(raw[:, 2:5]).astype(np.float32)
        output = {name: [] for name in ("x", "operating_settings", "rul", "engine_id", "cycle", "valid_mask")}
        for unit in np.unique(raw[:, 0]).astype(int):
            selected = raw[:, 0] == unit
            sensors, operating = normalized[selected], settings[selected]
            endpoints = range(1, len(sensors) + 1) if split == "train" else [len(sensors)]
            for end in endpoints:
                output["x"].append(prefix_window(sensors, end, length))
                output["operating_settings"].append(prefix_window(operating, end, length))
                label = len(sensors) - end if split == "train" else truth[unit]
                output["rul"].append(min(float(label), cap))
                output["engine_id"].append(unit)
                output["cycle"].append(end)
                output["valid_mask"].append(np.arange(length) >= max(0, length - end))
        prepared[split] = {name: np.asarray(values) for name, values in output.items()}
        prepared[split]["rul"] = prepared[split]["rul"].astype(np.float32)
    metadata = {
        "config": config,
        "normalization": "Training-fitted per-condition MinMax scaling; values are not clipped.",
        "condition_key": "First operating setting rounded to condition_decimals.",
        "sensor_scalers": {str(key): scaler_state(value) for key, value in scalers.items()},
        "operating_scaler": scaler_state(operating_scaler),
        "counts": {name: len(arrays["rul"]) for name, arrays in prepared.items()},
    }
    return prepared, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/cmapss.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = read_trajectories(args.raw_dir / f"train_{args.subset}.txt")
    test = read_trajectories(args.raw_dir / f"test_{args.subset}.txt")
    truth = np.loadtxt(args.raw_dir / f"RUL_{args.subset}.txt", ndmin=1)
    arrays, metadata = prepare_windows(train, test, truth, config)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, values in arrays.items():
        np.savez_compressed(args.output / f"{args.subset}_{split}.npz", **values)
    metadata["subset"] = args.subset
    (args.output / f"{args.subset}_preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"subset": args.subset, "windows": metadata["counts"]}, indent=2))


if __name__ == "__main__":
    main()
