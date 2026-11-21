"""Convert the public UIT-VSFC parquet splits to canonical text/label JSONL."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def convert(source, target, limit=None):
    rows = pd.read_parquet(source, columns=["sentence", "sentiment"])
    if limit is not None and limit < len(rows):
        rows, _ = train_test_split(rows, train_size=limit, stratify=rows.sentiment, random_state=7)
    with target.open("w", encoding="utf-8") as stream:
        for text, label in rows.itertuples(index=False, name=None):
            stream.write(json.dumps({"text": text, "label": int(label)}, ensure_ascii=False) + "\n")
    print(f"{target}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    convert(args.train, args.out / "train.jsonl", args.limit)
    convert(args.valid, args.out / "valid.jsonl")


if __name__ == "__main__":
    main()
