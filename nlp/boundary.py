"""Sentence style classifier + one-change decoder; train on original char offsets."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .blocks.changepoint import sentence_spans, locate_boundary, boundary_score


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fixture(count, prefix):
    rows = []
    for i in range(count):
        human = "He closed the wooden door. Rain tapped upon the glass. " * (1 + i % 3)
        machine = "Furthermore, it is essential to consider this comprehensive perspective. This highlights the significance of the situation. "
        rows.append({"id": f"{prefix}-{i}", "text": human + machine, "boundary_char_index": len(human)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="JSONL id,text; boundary in row or separate --labels")
    parser.add_argument("--labels", type=Path, help="IOAI train answers.jsonl")
    parser.add_argument("--valid", type=Path, help="JSONL inference rows; labels optional")
    parser.add_argument("--valid-labels", type=Path, help="Optional labels used only AFTER prediction")
    parser.add_argument("--out", type=Path, default=Path("runs/boundary"))
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    train, valid = (read_jsonl(args.train), read_jsonl(args.valid)) if args.train else (fixture(16, "train"), fixture(8, "valid"))
    if args.labels:
        labels_by_id = {r["id"]: r["boundary_char_index"] for r in read_jsonl(args.labels)}
        train = [dict(r, boundary_char_index=labels_by_id[r["id"]]) for r in train]
    if {r["id"] for r in train} & {r["id"] for r in valid}:
        parser.error("Train and inference IDs overlap")
    texts, labels = [], []
    for row in train:
        for start, end in sentence_spans(row["text"]):
            boundary = row["boundary_char_index"]
            if start >= boundary or end <= boundary:
                texts.append(row["text"][start:end])
                labels.append(int(start >= boundary))
    model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=30000),
                          LogisticRegression(max_iter=500, class_weight="balanced", random_state=7))
    model.fit(texts, labels)
    outputs = []
    for row in valid:
        spans = sentence_spans(row["text"])
        if len(spans) < 2:
            predicted = len(row["text"]) // 2
        else:
            p = model.predict_proba([row["text"][a:b] for a, b in spans])[:, 1]
            predicted = locate_boundary(p, [a for a, _ in spans])
        outputs.append({"id": row["id"], "boundary_char_index": predicted})
    if len({r["id"] for r in outputs}) != len(outputs):
        parser.error("Inference IDs must be unique")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in outputs), encoding="utf-8")
    joblib.dump(model, args.out / "model.joblib")
    valid_labels = {r["id"]: r["boundary_char_index"] for r in read_jsonl(args.valid_labels)} if args.valid_labels else {r["id"]: r["boundary_char_index"] for r in valid if "boundary_char_index" in r}
    if len(valid_labels) == len(outputs):
        print({"data": "jsonl" if args.train else "synthetic_fixture", "boundary_score_0_to_1": boundary_score(
            [r["boundary_char_index"] for r in outputs], [valid_labels[r["id"]] for r in outputs])})


if __name__ == "__main__":
    main()
