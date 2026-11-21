"""Search the curated registry without internet or ML dependencies."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", choices=["ml", "nlp", "vision", "vision_ml", "audio"])
    parser.add_argument("--priority", choices=["A", "B", "C"])
    parser.add_argument("--query", default="")
    args = parser.parse_args()
    rows = json.loads(Path(__file__).with_name("workshops_2026.json").read_text(encoding="utf-8"))["entries"]
    for row in rows:
        if args.area and row["area"] != args.area:
            continue
        if args.priority and row["priority"] != args.priority:
            continue
        if args.query.casefold() not in json.dumps(row, ensure_ascii=False).casefold():
            continue
        print(f"[{row['priority']}] {row['title']} | {row['event']} | docs/workshops/{row['id']}.md")


if __name__ == "__main__":
    main()
