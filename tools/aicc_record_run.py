"""Record an observed Cosmos smoke run; other AICC metrics need adapters."""
import argparse
import csv
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]


def validate_report(report):
    if report.get("status") != "completed":
        raise ValueError("A completed run report is required")
    experiments = report.get("experiments", [])
    if not experiments:
        raise ValueError("The report contains no measured experiments")
    for record in experiments:
        for key in ("raw_mean_kendall", "score", "seconds"):
            value = record.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Missing or non-finite measured value: " + key)
        if not -1 <= record["raw_mean_kendall"] <= 1 or not 0 <= record["score"] <= 1:
            raise ValueError("Kendall score is outside its valid range")
        if record.get("split") == "unseen_biome_holdout" and record.get("overlapping_biomes") != 0:
            raise ValueError("Biome holdout contains overlapping biomes")
    checks = report.get("submission_checks", [])
    if not checks or not all(c.get("valid_permutations") is True and c.get("rows", 0) > 0 for c in checks):
        raise ValueError("Submission checks did not pass")
    selected = [r for r in experiments if r["variant"] == report.get("selected_variant")
                and r["split"] == report.get("selection_split")]
    if not selected:
        raise ValueError("Selected model has no matching validation evidence")


def validate_notebook_url(value):
    parsed = urlparse(value)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme != "https" or parsed.netloc != "www.kaggle.com" or len(parts) != 3 or parts[0] != "code":
        raise ValueError("Expected a canonical Kaggle notebook URL")
    query = parse_qs(parsed.query)
    if parsed.fragment or set(query) - {"scriptVersionId"}:
        raise ValueError("Notebook URL must not contain session or credential parameters")
    return value


def record_run(report_path, challenge, notebook_url, version, public_score=None, private_score=None):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    validate_report(report)
    validate_notebook_url(notebook_url)
    if version < 1:
        raise ValueError("Version must be positive")
    queue_path = ROOT / "experiments/aicc/queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    matching = [item for item in queue["items"] if item["id"] == challenge]
    if len(matching) != 1:
        raise ValueError("Challenge must exist exactly once in the queue")
    if report.get("competition") != matching[0].get("competition_slug"):
        raise ValueError("Run report competition does not match the queued challenge")
    for score in (public_score, private_score):
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Observed score must be finite and in [0, 1]")
    destination = ROOT / "experiments/aicc" / challenge
    destination.mkdir(parents=True, exist_ok=True)
    evidence = {"notebook_url": notebook_url, "version": version, "run_status": "completed",
                "report_source": "Executed notebook stdout: SMOKE_RESULT_JSON",
                "public_score": public_score, "private_score": private_score,
                "phase": "smoke", "unattended_service_installed": False}
    (destination / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (destination / "run_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    with (destination / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["experiments"][0]))
        writer.writeheader()
        writer.writerows(report["experiments"])
    matching[0].update(status="completed", phase="smoke", notebook_url=notebook_url, version=version,
                       public_score=public_score, private_score=private_score,
                       leaderboard_score=private_score if private_score is not None else public_score,
                       report_path=challenge + "/REPORT.md")
    queue_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--notebook-url", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--private-score", type=float)
    args = parser.parse_args()
    print(json.dumps(record_run(args.report, args.challenge, args.notebook_url, args.version,
                                args.public_score, args.private_score), indent=2))


if __name__ == "__main__":
    main()

