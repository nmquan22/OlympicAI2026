"""Seasonal-naive versus ridge multi-step forecasting, with a purged time split."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .blocks.timeseries import forecast_windows, seasonal_naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, help="NPY [time,channels], regularly sampled, no NaNs")
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("runs/forecast"))
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    values = np.load(args.values) if args.values else (np.sin(np.arange(500) * 2 * np.pi / 24) + rng.normal(0, 0.1, 500))[:, None]
    x, y, origins = forecast_windows(values, args.context, args.horizon)
    cutoff = int(len(values) * 0.8)
    # Every train target must finish before the validation timestamp.
    train, valid = origins + args.horizon <= cutoff, origins >= cutoff
    if not train.any() or not valid.any():
        parser.error("Not enough data for this context/horizon and time split")
    flat = x.reshape(len(x), -1)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(flat[train], y[train].reshape(train.sum(), -1))
    prediction = model.predict(flat[valid]).reshape(y[valid].shape)
    naive = seasonal_naive(x[valid], args.horizon, args.period)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "predictions.npz", prediction=prediction, seasonal_naive=naive,
             target=y[valid], origin=origins[valid])
    print({"data": "npy" if args.values else "synthetic_fixture", "ridge_mae": float(np.abs(prediction - y[valid]).mean()),
           "seasonal_naive_mae": float(np.abs(naive - y[valid]).mean())})


if __name__ == "__main__":
    main()
