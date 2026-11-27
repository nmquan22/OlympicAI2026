"""Generate a reviewable Kaggle notebook with immutable source provenance."""
import argparse,json,re
from pathlib import Path
MODULES={'moment-match','lost-interpreter','word-lookups','inverse-translation','alchemy'}
def generate(task,runs,commit,dest):
 assert task in MODULES and runs and all(isinstance(i,int) and i>0 for i in runs)
 assert re.fullmatch('[0-9a-f]{40}',commit)
 prefix=''
 if task=='inverse-translation':
  prefix='import subprocess, sys\nsubprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1"], check=True)\n'
 code=prefix+f'''# Post-contest practice. Every experiment has fresh model/optimizer state.
SOURCE_COMMIT = {commit!r}
from urllib.request import urlopen
source = urlopen('https://raw.githubusercontent.com/nmquan22/OlympicAI2026/' + SOURCE_COMMIT + '/experiments/nlp_aicc/code/batch.py').read().decode('utf-8')
exec(compile(source, 'batch.py', 'exec'), globals())
manifest = run_batch({task!r}, {runs!r}, SOURCE_COMMIT)
'''
 nb=dict(cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(True))],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(nb,indent=1)+'\n')
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--runs',required=True,type=int,nargs='+');p.add_argument('--commit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 generate(a.task,a.runs,a.commit,a.output)
 print(json.dumps(dict(notebook=a.output,task=a.task,independent_experiments=len(a.runs))))
