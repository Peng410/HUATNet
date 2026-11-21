# HUATNet

Research utilities for **Temporal--Context State Organization for Uncertainty-Aware Aero-Engine RUL Prediction**.

## Research Overview

HUATNet studies probabilistic remaining useful life (RUL) prediction for aero-engines under varying operating conditions and observation disturbances. The approach combines temporal degradation information with contextual operating responses, using uncertainty-guided information interaction and local evolution regularization to support lifetime estimation. The study evaluates prediction accuracy, uncertainty characterization, and noise robustness on C-MAPSS and N-CMAPSS.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `huatnet_tools/framework.py` | Temporal and context branch interfaces and probabilistic RUL output |
| `huatnet_tools/losses.py` | Differentiable asymmetric regression, dynamics residual, Gaussian KL, and joint objective |
| `huatnet_tools/prepare.py` | C-MAPSS sensor selection, training-fitted normalization, and window preparation |
| `huatnet_tools/evaluate.py` | RMSE, MAE, NASA Score, interval coverage and width, and uncertainty-error correlation |
| `huatnet_tools/plot.py` | RUL trajectories, prediction parity, and uncertainty-error visualizations |
| `configs/cmapss.json` | Window length, RUL cap, selected sensors, and operating-group definition |

`HUATNetScaffold` accepts two encoders and a prediction function. It routes
aligned sensor windows and operating settings through the branches and returns
a RUL estimate and predictive standard deviation in cycles.

The loss functions accept prediction tensors, derivative and evolution tensors,
and Gaussian distribution parameters. The joint objective combines asymmetric
regression, the squared dynamics residual, and a KL term with linear warm-up.

## Installation

Use Python 3.10 or later. Run commands from the repository root.

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the official C-MAPSS text files in a local directory.

```text
datasets/CMAPSSData/
  train_FD004.txt
  test_FD004.txt
  RUL_FD004.txt
```

```bash
python -m huatnet_tools.prepare --raw-dir datasets/CMAPSSData --subset FD004 --output data/FD004
```

The default configuration selects 14 sensors and constructs windows of 50 observations. Sensor values are normalized within operating groups defined by the first operating setting rounded to one decimal place. All scaling statistics are fitted on the official training partition. The three operating settings use a separate training-fitted scaler.

Training windows advance by one observation. Short histories repeat the first observation on the left, with a corresponding valid-observation mask. Each test engine contributes its final observed window. RUL labels are capped at 125 cycles. Preprocessing statistics are saved in a JSON file alongside the window arrays.

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | `(samples, 50, 14)` | Normalized sensor observations |
| `operating_settings` | `(samples, 50, 3)` | Normalized operating settings |
| `rul` | `(samples,)` | RUL labels in cycles |
| `engine_id` | `(samples,)` | Engine identifiers within the source partition |
| `cycle` | `(samples,)` | Window endpoints |
| `valid_mask` | `(samples, 50)` | Observed positions marked as true |

## Prediction Evaluation

Provide predictions in a CSV file. The same evaluation interface accepts C-MAPSS and N-CMAPSS predictions expressed in cycles.

| Column | Use |
| --- | --- |
| `y_true`, `y_pred` | Required reference and predicted RUL |
| `lower`, `upper` | Optional prediction interval bounds, supplied together |
| `predictive_sd` | Optional predictive standard deviation |
| `engine_id`, `cycle` | Optional identifiers for trajectory visualization |

```bash
python -m huatnet_tools.evaluate --csv results/predictions.csv --output results/metrics.json
```

RMSE and MAE are reported in cycles. NASA Score is the sum of asymmetric penalties, using denominators 13 for underestimation and 10 for overestimation. PICP measures interval coverage, while MPIW measures interval width in cycles. Pearson correlation compares predictive standard deviation with absolute RUL error. Constant inputs produce a JSON `null` correlation.

## Visualization

```bash
python -m huatnet_tools.plot --csv results/lifecycle.csv --kind trajectory --engine 135 --output figures/trajectory.pdf
python -m huatnet_tools.plot --csv results/predictions.csv --kind parity --output figures/parity.pdf
python -m huatnet_tools.plot --csv results/predictions.csv --kind uncertainty-error --output figures/uncertainty_error.pdf
```

Trajectory plots display the supplied observations, predictions, and interval bounds in cycle order. Plotting supports PDF, SVG, and 300-dpi PNG output.

## License

The repository utilities are distributed under the MIT License.
