"""Record measured fields copied from visible Kaggle output when export times out.

This never fabricates an executed notebook or a complete stdout log. Evidence
explicitly says which fields were retained and links to the original version.
"""
import json,math,re
from pathlib import Path
from urllib.parse import urlparse
from nlp_record_run import ROOT
from nlp_status import render_status

def record_observation(report,url,version,script_id,notes):
 parsed=urlparse(url)
 assert parsed.scheme=='https' and parsed.netloc=='www.kaggle.com' and parsed.path.startswith('/code/shareaccc3a/')
 assert report['status']=='completed' and 'T4' in report['device']
 assert re.fullmatch(r'R\d{2,}',report['run_id']) and int(report['run_id'][1:])>0
 assert math.isfinite(report['validation']['value'])
 checks=report['submission_checks'];assert checks['ids_unique'] is True and checks['schema_valid'] is True and checks['row_count']>0
 base=ROOT/'experiments/nlp_aicc';queue=json.loads((base/'queue.json').read_text())
 item=next(t for t in queue['items'] if t['id']==report['task'])
 row=next((r for r in item['runs'] if r['run_id']==report['run_id']),None)
 if row is None:
  row=dict(run_id=report['run_id'],idea=report['config']['idea']);item['runs'].append(row)
 out=base/report['task']/'runs'/report['run_id'];out.mkdir(parents=True,exist_ok=True)
 report['capture_note']='Measured summary fields read from rendered Kaggle output; not a complete exported notebook/stdout.'
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'evidence.json').write_text(json.dumps(dict(notebook_url=url,version=version,script_version_id=script_id,status='completed',evidence_method='observed_rendered_kaggle_output',full_notebook_export_available=False),indent=2)+'\n')
 (out/'LEARNED.md').write_text(notes+'\n\nBằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.\n')
 row.update(status='completed',notebook_version=version,metric=report['validation'],report_path=f"{report['task']}/runs/{report['run_id']}/report.json")
 required=[r for r in item['runs'] if int(r['run_id'][1:])<=7]
 item['status']='completed' if len(required)==7 and all(r['status']=='completed' for r in required) else 'running_experiments'
 item['notebook_url']=parsed._replace(query='',fragment='').geturl();item.pop('last_error',None)
 (base/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n');render_status()
 return list(out.glob('*'))+[base/'queue.json',base/'STATUS.md']
