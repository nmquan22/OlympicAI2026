"""Validate IDs before writing. Labels are supplied separately from inference data."""
import csv
import json
from pathlib import Path


def write_predictions(path, ids, values, value_name="prediction"):
    ids, values = list(ids), list(values)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("Need exactly one prediction per unique ID")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", value_name])
        writer.writerows(zip(ids, values))


def order_to_ranks(order):
    """[chunk IDs in time order] -> ranks indexed by shuffled chunk ID (IOAI 2026)."""
    if sorted(order) != list(range(len(order))):
        raise ValueError("Order must be a permutation of 0..n-1")
    ranks = [0] * len(order)
    for rank, chunk in enumerate(order):
        ranks[chunk] = rank
    return ranks


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
