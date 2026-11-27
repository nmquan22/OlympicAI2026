"""Measure mean and TF-IDF ridge controls on the official Subtask 1 dev set."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from prepare_data import task1_splits


def metrics(target, pred):
    error = np.asarray(pred) - np.asarray(target)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    # Official scorer divides the summed V and A squared errors by N, not 2N.
    official_rmse = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    return {"official_RMSE_VA": official_rmse, "joint_coordinate_RMSE": official_rmse / 2 ** 0.5,
            "RMSE_V": float(rmse[0]), "RMSE_A": float(rmse[1])}


def context(example, local=False):
    text, aspect = example["text"].lower(), example["aspect"].lower()
    if local:
        match = re.search(re.escape(aspect), text)
        if match is not None:
            text = text[max(0, match.start() - 70): match.end() + 70]
    return text


def fit_features(train, dev, local):
    x_train, x_dev = [], []
    columns = [(lambda e: context(e, local), "word", (1, 2)),
               (lambda e: context(e, local), "char_wb", (3, 5)),
               (lambda e: e["aspect"].lower(), "word", (1, 2))]
    for convert, analyzer, ngram_range in columns:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2,
                              max_features=40000, sublinear_tf=True, dtype=np.float64)
        x_train.append(vec.fit_transform([convert(e) for e in train]))
        x_dev.append(vec.transform([convert(e) for e in dev]))
    return hstack(x_train).tocsr(), hstack(x_dev).tocsr()


def export_predictions(rows, examples, predictions, output):
    mapping = {(e["id"], e["aspect"]): p for e, p in zip(examples, predictions)}
    lines = []
    for row in rows:
        items = []
        for item in row["Aspect_VA"]:
            v, a = mapping[row["ID"], item["Aspect"]]
            items.append({"Aspect": item["Aspect"], "VA": f"{v:.2f}#{a:.2f}"})
        lines.append(json.dumps({"ID": row["ID"], "Aspect_VA": items}, ensure_ascii=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train, dev, dev_rows, removed = task1_splits(args.data)
    y_train = np.array([e["target"] for e in train])
    y_dev = np.array([e["target"] for e in dev])
    report = {"status": "completed_cpu_controls", "pretrained_weights": False, "test_evaluated": False,
              "train_aspects": len(train), "dev_aspects": len(dev), "overlapping_train_rows_removed": removed, "runs": []}
    mean_pred = np.round(np.repeat(y_train.mean(0)[None], len(dev), axis=0), 2)
    report["runs"].append({"run": "C00", "idea": "Train target mean", **metrics(y_dev, mean_pred)})
    export_predictions(dev_rows, dev, mean_pred, args.out / "C00_dev.jsonl")
    for run, local in (("C01", False), ("C02", True)):
        start = time.monotonic()
        x_train, x_dev = fit_features(train, dev, local)
        model = Ridge(alpha=3.0, solver="lsqr").fit(x_train, y_train)
        pred = np.clip(model.predict(x_dev), 1.0, 9.0)
        # Report exactly the two-decimal predictions accepted by the task.
        pred = np.round(pred, 2)
        report["runs"].append({"run": run, "idea": "Local context TF-IDF + ridge" if local else "Full context TF-IDF + ridge",
                               "alpha": 3.0, "features": x_train.shape[1], "seconds": time.monotonic() - start,
                               **metrics(y_dev, pred)})
        export_predictions(dev_rows, dev, pred, args.out / f"{run}_dev.jsonl")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
