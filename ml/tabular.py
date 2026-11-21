"""Leakage-aware tabular baseline. Real data use separate train/valid CSV files."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error

from common.submission import write_predictions
from .blocks.tabular import tabular_pipeline


def fixture(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, 2))
    return pd.DataFrame({"id": [f"{seed}-{i}" for i in range(count)], "x1": x[:, 0], "x2": x[:, 1],
                         "region": rng.choice(["north", "south"], count), "target": (x.sum(1) > 0).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--predict", type=Path, help="Optional unlabelled CSV")
    parser.add_argument("--target", default="target")
    parser.add_argument("--id", default="id")
    parser.add_argument("--numeric", nargs="*", default=["x1", "x2"])
    parser.add_argument("--categorical", nargs="*", default=["region"])
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--model", choices=["linear", "trees"], default="linear")
    parser.add_argument("--out", type=Path, default=Path("runs/tabular"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    if args.id in args.numeric + args.categorical or args.target in args.numeric + args.categorical:
        parser.error("ID and target cannot be feature columns")
    train, valid = (pd.read_csv(args.train), pd.read_csv(args.valid)) if args.train else (fixture(7, 100), fixture(8, 40))
    if set(train[args.id]).intersection(valid[args.id]):
        parser.error("Train and validation IDs overlap")
    model = tabular_pipeline(args.numeric, args.categorical, task=args.task, model=args.model)
    model.fit(train, train[args.target])
    prediction = model.predict(valid)
    metrics = {"macro_f1": float(f1_score(valid[args.target], prediction, average="macro"))} if args.task == "classification" else {"mae": float(mean_absolute_error(valid[args.target], prediction))}
    data = pd.read_csv(args.predict) if args.predict else valid
    write_predictions(args.out / "predictions.csv", data[args.id], model.predict(data))
    joblib.dump(model, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({"data": "csv" if args.train else "synthetic_fixture", **metrics})


if __name__ == "__main__":
    main()
