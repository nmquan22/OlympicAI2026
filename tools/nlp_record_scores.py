"""Persist scores observed in Kaggle UI, keeping source-version identity explicit."""
import csv,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'experiments/nlp_aicc'
def record(task, rows, metric, direction, url):
    queue=json.loads((BASE/'queue.json').read_text())
    item=next(x for x in queue['items'] if x['id']==task)
    target=BASE/task/'SCORES.csv'
    previous={}
    if target.exists():
        previous={r['run']:r for r in csv.DictReader(target.open())}
    for row in rows:
        run=row['run'];public=row['public'];private=row['private']
        fields=dict(leaderboard_public=public,leaderboard_private=private,
            leaderboard_metric=metric,leaderboard_direction=direction,
            score_precision='4 decimals shown by Kaggle UI',frozen_before_scoring=True,
            submission_evidence=dict(url=url,**row))
        report=BASE/task/'runs'/run/'report.json'
        data=json.loads(report.read_text());data.update(fields)
        report.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        next(r for r in item['runs'] if r['run_id']==run).update(fields)
        previous[run]=row
    with target.open('w') as f:
        out=csv.DictWriter(f,fieldnames=['run','public','private','version_id','output_file','description_note'])
        out.writeheader();out.writerows(previous[k] for k in sorted(previous))
    (BASE/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('input');args=p.parse_args()
    for task, data in json.loads(Path(args.input).read_text()).items():record(task,**data)
