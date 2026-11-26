"""Import a completed T4 experiment from an executed Kaggle notebook."""
import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT=Path(__file__).resolve().parents[1]


def read_executed_notebook(path):
    notebook=json.loads(Path(path).read_text())
    chunks=[]
    for cell in notebook["cells"]:
        cell.get("metadata",{}).pop("execution",None)
        for output in cell.get("outputs",[]):
            text=output.get("text","")
            chunks.append("".join(text) if isinstance(text,list) else text)
    log="\n".join(chunks)
    lines=[line for line in log.splitlines() if line.startswith("NLP_RUN_RESULT ")]
    if len(lines)!=1:
        raise ValueError("Expected exactly one measured run report in executed notebook stdout")
    report=json.loads(lines[0].split(" ",1)[1])
    if report.get("status")!="completed" or "T4" not in report.get("device",""):
        raise ValueError("Only successfully completed T4 runs count")
    if not math.isfinite(report["validation"]["value"]):
        raise ValueError("Validation metric must be finite")
    checks=report.get("submission_checks",{})
    if checks.get("ids_unique") is not True or checks.get("schema_valid") is not True or checks.get("row_count",0)<=0:
        raise ValueError("Submission checks did not pass")
    if report.get("run_id") not in {f"R{i:02d}" for i in range(1,8)}:
        raise ValueError("Unexpected run identifier")
    return notebook,log,report


def record_run(notebook_path,url,version,notes):
    parsed=urlparse(url);parts=parsed.path.strip("/").split("/")
    query=parse_qs(parsed.query)
    if parsed.scheme!="https" or parsed.netloc!="www.kaggle.com" or len(parts)!=3 or parts[0]!="code":
        raise ValueError("Expected a canonical Kaggle notebook URL")
    if parsed.fragment or set(query)-{"scriptVersionId"} or len(query.get("scriptVersionId",[]))!=1:
        raise ValueError("Use the observed version-specific URL without session parameters")
    notebook,log,report=read_executed_notebook(notebook_path)
    queue_file=ROOT/"experiments/nlp_aicc/queue.json"
    queue=json.loads(queue_file.read_text())
    item=next(x for x in queue["items"] if x["id"]==report["task"])
    row=next(x for x in item["runs"] if x["run_id"]==report["run_id"])
    dest=queue_file.parent/report["task"]/"runs"/report["run_id"]
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"notebook.ipynb").write_text(json.dumps(notebook,indent=1)+"\n")
    (dest/"run.log").write_text(log)
    (dest/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    evidence=dict(notebook_url=url,version=version,script_version_id=int(query["scriptVersionId"][0]),status="completed")
    (dest/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    (dest/"LEARNED.md").write_text(notes)
    row.update(status="completed",notebook_version=version,metric=report["validation"],
               report_path=f"{report['task']}/runs/{report['run_id']}/report.json")
    item.pop("last_error",None)
    item["status"]="completed" if all(r["status"]=="completed" for r in item["runs"]) else "running_experiments"
    item["notebook_url"]=parsed._replace(query="",fragment="").geturl()
    queue_file.write_text(json.dumps(queue,indent=2,ensure_ascii=False)+"\n")
    from nlp_status import render_status
    render_status()
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook",required=True,type=Path)
    p.add_argument("--url",required=True)
    p.add_argument("--version",required=True,type=int)
    p.add_argument("--notes",required=True,type=Path)
    args=p.parse_args()
    report=record_run(args.notebook,args.url,args.version,args.notes.read_text())
    print(json.dumps({"task":report["task"],"run":report["run_id"],"metric":report["validation"]}))

