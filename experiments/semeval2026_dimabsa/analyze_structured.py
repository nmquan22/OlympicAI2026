"""Separate candidate coverage, decoding, category and VA errors without training."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_data import read_jsonl


def pair(item):
    return item["Aspect"].lower(), item["Opinion"].lower()


def span_statistics(gold_sets, predicted_sets):
    gold = sum(map(len, gold_sets))
    predicted = sum(map(len, predicted_sets))
    matched = sum(len(g & p) for g, p in zip(gold_sets, predicted_sets))
    return {"gold": gold, "predicted": predicted, "matched": matched,
            "precision": matched / max(1, predicted), "recall": matched / max(1, gold),
            "F1": 2 * matched / max(1, gold + predicted)}


def analyze(rows, predictions, candidates, category_names):
    gold_pairs = [{pair(q) for q in row["Quadruplet"]} for row in rows]
    predicted_pairs = [{pair(q) for q in row["Quadruplet"]} for row in predictions]
    candidate_pairs = [{pair(q) for q in sentence} for sentence in candidates]
    result = {"pair_decoding": span_statistics(gold_pairs, predicted_pairs),
              "top8_candidate_pair_coverage": span_statistics(gold_pairs, candidate_pairs)}
    for field in ("Aspect", "Opinion"):
        gold_spans = [{q[field].lower() for q in row["Quadruplet"]} for row in rows]
        predicted_spans = [{q[field].lower() for q in row["Quadruplet"]} for row in predictions]
        candidate_spans = [{q[field].lower() for q in sentence} for sentence in candidates]
        result[field.lower() + "_decoding"] = span_statistics(gold_spans, predicted_spans)
        result[field.lower() + "_candidate_coverage"] = span_statistics(gold_spans, candidate_spans)
    matched, category_correct, va_errors, unseen = 0, 0, [], []
    examples = []
    for row, predicted in zip(rows, predictions):
        lookup = {pair(q): q for q in predicted["Quadruplet"]}
        for gold in row["Quadruplet"]:
            if gold["Category"] not in category_names:
                unseen.append({"ID": row["ID"], "category": gold["Category"]})
            if pair(gold) in lookup:
                pred = lookup[pair(gold)]
                matched += 1
                category_correct += pred["Category"].lower() == gold["Category"].lower()
                y = np.array(list(map(float, gold["VA"].split("#"))))
                p = np.array(list(map(float, pred["VA"].split("#"))))
                va_errors.append(float(np.sum((p - y) ** 2)))
            elif len(examples) < 8:
                examples.append({"ID": row["ID"], "text": row["Text"], "missed_gold": gold})
    result["category_given_correct_pair"] = {"matched_pairs": matched, "category_correct": category_correct,
                                              "accuracy": category_correct / max(1, matched)}
    result["VA_given_correct_pair_RMSE_VA"] = float(np.sqrt(np.mean(va_errors))) if va_errors else None
    result["unseen_train_categories_in_gold"] = unseen
    result["first_missed_examples"] = examples
    result["interpretation"] = "Candidate coverage is an upper bound for pair recall at fixed top-8 pruning; conditional VA/category metrics ignore missing pairs."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    analysis = analyze(read_jsonl(args.gold), read_jsonl(args.run / "dev_task3_predictions.jsonl"),
                       json.loads((args.run / "dev_candidates.json").read_text()), report["categories"])
    (args.run / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
