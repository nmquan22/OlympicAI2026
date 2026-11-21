"""A train-only TF-IDF baseline for short text classification."""
import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, make_pipeline

from .blocks.data import read_jsonl
from .train import fixture


def text_baseline():
    features = FeatureUnion([
        ("character", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(class_weight="balanced", max_iter=500))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--valid", type=Path)
    args = parser.parse_args()
    if bool(args.train) != bool(args.valid):
        parser.error("Supply both --train and --valid")
    train, valid = (read_jsonl(args.train), read_jsonl(args.valid)) if args.train else fixture()
    model = text_baseline().fit([r["text"] for r in train], [r["label"] for r in train])
    pred = model.predict([r["text"] for r in valid])
    print(classification_report([r["label"] for r in valid], pred, zero_division=0, digits=4))


if __name__ == "__main__":
    main()
