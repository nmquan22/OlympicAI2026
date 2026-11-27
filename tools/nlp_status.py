"""Generate a readable status page from measured run reports, without inventing scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/nlp_aicc'
def render_status():
 queue=json.loads((BASE/'queue.json').read_text())
 rows=[];total=0
 for task in queue['items']:
  completed=[r for r in task['runs'] if r['status']=='completed'];total+=len(completed)
  metric='Chưa có kết quả'
  if completed and completed[0]['metric']['metric']=='BPE_LM_NLL_proxy':
   metric='Chưa xếp hạng segmentation: cần Boundary F1'
  elif completed:
   best=sorted(completed,key=lambda r:r['metric']['value'],reverse=completed[0]['metric']['direction']=='maximize')[0]
   m=best['metric'];metric=f"{best['run_id']}: {m['value']:.5f} {m['metric']}"
  rows.append(f"| [{task['title']}]({task['id']}/IDEAS.md) | {len(completed)}/7 | {metric} |")
 lines=['# Tiến độ thực nghiệm NLP', '',f'**{total} thí nghiệm T4 thành công đã lưu bằng chứng; mục tiêu tối thiểu 35 (7/bài).** Số thí nghiệm khác số Saved Version; lượt lỗi không được tính. Tối đa hai Saved Version đang chạy.', '', '| Bài | Đã hoàn tất | Tốt nhất trong điều kiện đã chạy |','|---|---:|---|',*rows,'', 'Accuracy/MAE trên validation, LM NLL proxy và official inverse score là các metric khác nhau. Không so điểm ngang giữa các bài; không gọi NLL là Boundary F1.', '']
 for task in queue['items']:
  lines += [f"## {task['title']}",'','| Run | Ý tưởng | Kết quả | Nhật ký |','|---|---|---|---|']
  for r in task['runs']:
   m=r.get('metric');score=f"{m['value']:.5f} {m['metric']}" if m else r['status']
   link=f"[Đọc]({task['id']}/runs/{r['run_id']}/LEARNED.md)" if r['status']=='completed' else 'Chưa có kết quả'
   lines.append(f"| {r['run_id']} | {r['idea']} | {score} | {link} |")
  lines.append('')
 (BASE/'STATUS.md').write_text('\n'.join(lines))
 return total
if __name__=='__main__':print(render_status())
