"""Audit completed experiment evidence, unique run counts, and task metrics."""
import ast,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/nlp_aicc'

def audit():
 queue=json.loads((BASE/'queue.json').read_text());seen=set();by_task={};problems=[];scored=0
 for task in queue['items']:
  passed=[]
  for row in task['runs']:
   if row.get('status')!='completed':continue
   key=(task['id'],row['run_id'])
   if key in seen:problems.append(f'duplicate {key}')
   seen.add(key)
   folder=BASE/task['id']/'runs'/row['run_id']
   try:
    report=json.loads((folder/'report.json').read_text());evidence=json.loads((folder/'evidence.json').read_text())
    assert report['task']==task['id'] and report['run_id']==row['run_id']
    assert report['status']=='completed' and 'T4' in report['device']
    assert math.isfinite(report['validation']['value']) and report['validation']['samples']>0
    assert report['validation']['value']==row['metric']['value']
    assert re.fullmatch('[0-9a-f]{40}',report['source_commit'])
    assert report['elapsed_seconds']>0
    checks=report['submission_checks'];assert checks['ids_unique'] and checks['schema_valid'] and checks['row_count']>0
    assert evidence['version']>0 and evidence['script_version_id']>0
    assert evidence['notebook_url'].startswith('https://www.kaggle.com/code/shareaccc3a/')
    assert (folder/'LEARNED.md').stat().st_size>100
    if evidence.get('full_notebook_export_available') is False:assert 'capture_note' in report
    else:assert (folder/'notebook.ipynb').exists()
    if report.get('leaderboard_public') is not None:
     assert math.isfinite(report['leaderboard_public']) and math.isfinite(report['leaderboard_private'])
     assert report['leaderboard_public']==row['leaderboard_public'] and report['leaderboard_private']==row['leaderboard_private']
     assert report['frozen_before_scoring'] is True
     scored+=1
    if task['id']=='word-lookups':
     assert report['leaderboard_metric']=='Boundary F1'
     assert 0<=report['leaderboard_public']<=1 and 0<=report['leaderboard_private']<=1
     assert report['frozen_before_scoring'] is True
    passed.append(row['run_id'])
   except Exception as exc:problems.append(f'{task["id"]}/{row["run_id"]}: {type(exc).__name__}: {exc}')
  required={f'R{i:02d}' for i in range(1,8)}
  by_task[task['id']]=dict(successful_unique=len(passed),minimum_seven_met=required.issubset(passed))
 for path in list((BASE/'code').glob('*.py'))+[ROOT/'tools/nlp_record_observation.py',ROOT/'tools/nlp_status.py',ROOT/'tools/nlp_eta.py']:
  try:ast.parse(path.read_text())
  except SyntaxError as exc:problems.append(str(exc))
 return dict(success=not problems and all(x['minimum_seven_met'] for x in by_task.values()),
             successful_unique_experiments=len(seen),scored_unique_submissions=scored,by_task=by_task,problems=problems,
             notes=['Counts unique completed model/decoder configurations, not Saved Versions or retry attempts.',
                    'This is an evidence/schema audit; it does not replay GPU training or establish statistical significance.'])
if __name__=='__main__':
 result=audit();print(json.dumps(result,indent=2));raise SystemExit(0 if result['success'] else 1)
