"""Run independently initialized experiments inside one Kaggle saved version.

Each run keeps its original seed/split/budget and has an isolated output folder.
GPU experiments, notebook versions and failed attempts are counted separately.
"""
import gc
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen

MODULES={
 'moment-match':('moment_match','run_moment'),
 'lost-interpreter':('lost_interpreter','run_lost'),
 'word-lookups':('word_lookups','run_word'),
 'inverse-translation':('inverse_translation','run_inverse'),
 'alchemy':('alchemy','run_alchemy')}

def run_batch(task,run_ids,source_commit):
 module,entry=MODULES[task]
 base=f'https://raw.githubusercontent.com/nmquan22/OlympicAI2026/{source_commit}/experiments/nlp_aicc/code/'
 code={name:urlopen(base+name).read().decode('utf-8') for name in ['common.py',module+'.py']}
 root=Path('/kaggle/working')/task;root.mkdir(parents=True,exist_ok=True)
 # Verify every requested configuration before training the first experiment.
 preflight={'SOURCE_COMMIT':source_commit,'__name__':'__main__'}
 for name,source in code.items():exec(compile(source,name,'exec'),preflight)
 table_name={'moment-match':'MOMENT_CONFIGS','lost-interpreter':'LOST_CONFIGS','word-lookups':'WORD_CONFIGS','inverse-translation':'INVERSE_CONFIGS','alchemy':'ALCHEMY_CONFIGS'}[task]
 available=set(preflight[table_name])
 missing=set(run_ids)-available
 if missing:raise ValueError(f'Requested config IDs missing from immutable source {source_commit}: {sorted(missing)}; available={sorted(available)}')
 print('BATCH_PREFLIGHT '+json.dumps(dict(task=task,source_commit=source_commit,requested=run_ids,available=sorted(available))),flush=True)
 preflight.clear();del preflight
 manifest={'task':task,'source_commit':source_commit,'run_ids':run_ids,'completed':[],
           'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in code.items()}}
 for run_id in run_ids:
  # A new globals dictionary prevents model/optimizer state crossing experiments.
  ns={'SOURCE_COMMIT':source_commit,'__name__':'__main__'}
  for name,source in code.items():exec(compile(source,name,'exec'),ns)
  output=root/f'R{run_id:02d}';output.mkdir(parents=True,exist_ok=True)
  ns['OUT']=output
  tick=time.time();report=ns[entry](run_id)
  assert report['status']=='completed' and 'T4' in report['device']
  manifest['completed'].append({'run_id':report['run_id'],'metric':report['validation'],
        'elapsed_seconds':time.time()-tick,'output':str(output.relative_to('/kaggle/working'))})
  (root/'batch_manifest.json').write_text(json.dumps(manifest,indent=2))
  print('BATCH_PROGRESS '+json.dumps(manifest['completed'][-1]),flush=True)
  torch=ns['torch'];ns.clear();del ns,report;gc.collect();torch.cuda.empty_cache()
 print('BATCH_RESULT '+json.dumps(manifest),flush=True)
 return manifest
