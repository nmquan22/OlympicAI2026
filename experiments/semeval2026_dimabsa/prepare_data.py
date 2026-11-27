"""Download a pinned official English restaurant dataset and audit its splits."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

SOURCE_REPO = "DimABSA/DimABSA2026"
SOURCE_COMMIT = "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def aspect_examples(rows, training=False):
    """Convert training quads to one regression target per text/aspect pair.

    Multiple opinion annotations for an aspect are averaged. NULL aspects are
    excluded from this explicit-aspect exercise. Dev/test targets stay unchanged.
    """
    output = []
    for row in rows:
        groups = defaultdict(list)
        key = "Quadruplet" if training else "Aspect_VA"
        for item in row[key]:
            aspect = item["Aspect"]
            if aspect != "NULL":
                groups[aspect].append([float(v) for v in item["VA"].split("#")])
        for aspect, values in groups.items():
            target = [sum(v[k] for v in values) / len(values) for k in range(2)]
            output.append({"id": row["ID"], "text": row["Text"], "aspect": aspect, "target": target})
    return output


def task1_splits(root):
    root = Path(root)
    train_rows = read_jsonl(root / "eng_restaurant_train_alltasks.jsonl")
    dev_rows = read_jsonl(root / "eng_restaurant_dev_task1.jsonl")
    normalize = lambda text: " ".join(text.lower().split())
    dev_texts = {normalize(row["Text"]) for row in dev_rows}
    clean_train = [row for row in train_rows if normalize(row["Text"]) not in dev_texts]
    return aspect_examples(clean_train, training=True), aspect_examples(dev_rows), dev_rows, len(train_rows) - len(clean_train)


def audit(root):
    root = Path(root)
    files = []
    texts = {}
    for path in sorted(root.glob("*.jsonl")):
        rows = read_jsonl(path)
        key = next(k for k in ("Quadruplet", "Triplet", "Aspect_VA") if k in rows[0])
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "sentences": len(rows), "annotations": sum(len(r[key]) for r in rows),
                      "null_aspects": sum(q["Aspect"] == "NULL" for r in rows for q in r[key])})
        texts[path.name] = {" ".join(r["Text"].lower().split()) for r in rows}
    train_name = "eng_restaurant_train_alltasks.jsonl"
    overlaps = {name: len(values & texts[train_name]) for name, values in texts.items() if name != train_name}
    clean_train, _, _, removed = task1_splits(root)
    return {"source_repository": SOURCE_REPO, "source_commit": SOURCE_COMMIT, "files": files,
            "normalized_text_overlap_with_train": overlaps,
            "training_aspect_examples": len(aspect_examples(read_jsonl(root / train_name), training=True)),
            "clean_training_aspect_examples": len(clean_train), "overlapping_train_rows_removed": removed,
            "training_policy": "exclude train rows overlapping dev text; train-only vocabulary/features/weights; dev selection; test final evaluation only",
            "training_target_policy": "drop NULL aspect; mean VA over annotations with the same sentence ID and exact aspect"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = ["task-dataset/track_a/subtask_1/eng/eng_restaurant_train_alltasks.jsonl"]
    paths += [f"task-dataset/track_a/subtask_{k}/eng/eng_restaurant_{split}_task{k}.jsonl"
              for split in ("dev", "test") for k in (1, 2, 3)]
    if args.download:
        for path in paths:
            target = args.out / Path(path).name
            urlretrieve(f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{path}", target)
            target.write_text(target.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    result = audit(args.out)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
