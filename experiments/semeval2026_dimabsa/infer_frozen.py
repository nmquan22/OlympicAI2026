"""Frozen inference from this experiment's own scratch-trained checkpoints."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from cpu_baselines import export_predictions, fit_features
from prepare_data import read_jsonl, task1_splits
from train_scratch import AspectDataset, Config, ScratchRegressor, collate, predict
from train_structured import SpanPairModel, build_example, cache_predictions, collate as span_collate, decode


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = args.bundle / "data"
    train, dev, dev_rows, _ = task1_splits(data)
    y_train = np.array([row["target"] for row in train])

    r01 = args.bundle / "checkpoints/R01"
    checkpoint = torch.load(r01 / "best.pt", map_location="cpu", weights_only=True)
    cfg = Config(**checkpoint["config"])
    tokenizer = Tokenizer.from_file(str(r01 / "tokenizer.json"))
    model = ScratchRegressor(cfg, tokenizer.get_vocab_size(), y_train.mean(0)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    regression_counts = {}
    for split in ("dev", "test"):
        rows = dev_rows if split == "dev" else read_jsonl(data / "eng_restaurant_test_task1_inputs.jsonl")
        examples = dev if split == "dev" else [
            {"id": row["ID"], "text": row["Text"], "aspect": item["Aspect"], "target": [5., 5.]}
            for row in rows for item in row["Aspect_VA"]]
        loader = DataLoader(AspectDataset(examples, tokenizer, cfg.max_length), batch_size=32,
                            shuffle=False, collate_fn=collate)
        neural = predict(model, loader, device)
        # Match the two-decimal exported component files before the fixed average.
        neural = np.array([[float(f"{v:.2f}") for v in pair] for pair in neural])
        x_train, x_eval = fit_features(train, examples, local=True)
        ridge_model = Ridge(alpha=3.0, solver="lsqr").fit(x_train, y_train)
        ridge = np.round(np.clip(ridge_model.predict(x_eval), 1, 9), 2)
        ensemble = np.round((ridge + neural) / 2, 2)
        for name, values in (("R01", neural), ("C02", ridge), ("C03", ensemble)):
            export_predictions(rows, examples, values, args.out / f"{split}_task1_{name}.jsonl")
        regression_counts[split] = len(examples)
    del model

    s01 = args.bundle / "checkpoints/S01"
    checkpoint = torch.load(s01 / "best.pt", map_location="cpu", weights_only=True)
    report = json.loads((s01 / "report.json").read_text())
    category_names = checkpoint["categories"]
    tokenizer = Tokenizer.from_file(str(s01 / "tokenizer.json"))
    model = SpanPairModel(tokenizer.get_vocab_size(), len(category_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    category_map = {name: i for i, name in enumerate(category_names)}
    structured_counts = {}
    for split in ("dev", "test"):
        source = data / ("eng_restaurant_dev_task3.jsonl" if split == "dev" else "eng_restaurant_test_task3_inputs.jsonl")
        rows3 = [{"ID": row["ID"], "Text": row["Text"]} for row in read_jsonl(source)]
        examples = [build_example(row, tokenizer, category_map) for row in rows3]
        loader = DataLoader(examples, batch_size=16, shuffle=False, collate_fn=span_collate)
        cache = cache_predictions(model, loader, category_names, device)
        for task, field in ((2, "Triplet"), (3, "Quadruplet")):
            source = data / (f"eng_restaurant_dev_task{task}.jsonl" if split == "dev" else f"eng_restaurant_test_task{task}_inputs.jsonl")
            rows = read_jsonl(source)
            if [row["Text"] for row in rows] != [row["Text"] for row in rows3]:
                raise ValueError("Subtask text order differs")
            predictions = [decode(candidates, report["span_threshold"], report["link_threshold"], task) for candidates in cache]
            write_jsonl(args.out / f"{split}_task{task}_S01.jsonl",
                        [{"ID": row["ID"], field: prediction} for row, prediction in zip(rows, predictions)])
        structured_counts[split] = len(rows3)
    result = {"run": "E01", "operation": "inference only; no optimizer or model selection",
              "checkpoint_origin": "R01 v1 and S01 v2 scratch-trained on this task's clean train split",
              "task1_recipe": "0.5 C02 + 0.5 R01, rounded cached component predictions",
              "structured_best_epoch": report["best_epoch"],
              "structured_thresholds": {k: report[k] for k in ("span_threshold", "link_threshold")},
              "regression_aspects": regression_counts, "structured_sentences": structured_counts,
              "test_gold_present": False, "test_model_selection": False,
              "torch_version": torch.__version__, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "seconds": time.monotonic() - started}
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
