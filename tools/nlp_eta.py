"""Estimate the first useful status check from measured completed experiments.

No network polling and no credentials. Queue delay is deliberately not guessed.
"""
import argparse,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/nlp_aicc'

def estimate(task,runs,startup=60):
 records=[]
 for p in (BASE/task/'runs').glob('R*/report.json'):
  r=json.loads(p.read_text())
  if r.get('status')=='completed':records.append(r)
 if not records:raise ValueError('Measure a pilot run before estimating this task')
 measured={int(r['run_id'][1:]):r for r in records}
 seconds=[]
 for run in runs:
  if run in measured:value=measured[run]['elapsed_seconds']
  elif task=='moment-match' and run>=10:
   value=max(90,1.5*statistics.median(r['elapsed_seconds'] for r in records))
  else:value=max(r['elapsed_seconds'] for r in records)
  seconds.append(value)
 first_check=math.ceil((sum(seconds)+startup)/60)*60
 return dict(task=task,run_ids=runs,measured_code_seconds=[round(x,1) for x in seconds],startup_buffer_seconds=startup,
             first_check_after_seconds=first_check,first_check_after_minutes=first_check/60,
             queue_delay_included=False,queued_backoff_seconds=[120,240,480,900],max_concurrent_saved_versions=2)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--runs',nargs='+',required=True,type=int);p.add_argument('--startup',type=int,default=60);a=p.parse_args()
 print(json.dumps(estimate(a.task,a.runs,a.startup),ensure_ascii=False,indent=2))
