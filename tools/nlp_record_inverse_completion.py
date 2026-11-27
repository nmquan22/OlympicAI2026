"""Persist the measured V05 Inverse results and verified late-submission scores."""
import json,csv
from pathlib import Path
from nlp_record_observation import record_observation
from nlp_record_scores import record
from nlp_status import render_status
from nlp_audit import audit
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/nlp_aicc'
COMMIT='84e9a9efaa6346578162fa12915a8368f18e6485'
URL='https://www.kaggle.com/code/shareaccc3a/notebook28ae3a7683?scriptVersionId=374816257'

def main():
    scores=[]
    for number,score,exact,similarity,elapsed,peak,probe,candidates,calls,grad,public in [
        (8,97.27777777777777,23/24,.9944444444444445,15.156506538391113,331.4501953125,3.801254987716675,3083,90,2,94.5555),
        (9,100.,1.,1.,8.28272008895874,340.1103515625,3.6464951038360596,2600,85,16,100.)]:
        cfg=dict(idea='Top8 exact search with contextual HotFlip fallback' if number==8 else 'Contextual gradient coordinate search from top1 probe seeds',
            topk=8,loss='margin',search='grid' if number==8 else 'gradient',restart=0,gradient=True,gradient_topk=16,gradient_steps=16 if number==8 else 24,
            probe_batch=512,candidate_batch=64,budget_seconds=720)
        report=dict(task='inverse-translation',run_id=f'R{number:02d}',config=cfg,seed=42,device='Tesla T4',gpu_count=2,source_commit=COMMIT,
            versions=dict(torch='2.11.0+cu130',numpy='2.1.3',pandas='2.3.3',**{'scikit-learn':'1.6.1','tokenizers':'0.22.2','transformers':'4.57.1'}),
            status='completed',elapsed_seconds=elapsed,validation=dict(metric='official_score_0_100',value=score,direction='maximize',samples=24,split='explicit_public_targets',
                official=dict(score=score,exact_rate=exact,mean_similarity=similarity,valid_rate=1.)),
            submission_checks=dict(ids_unique=True,schema_valid=True,row_count=24,token_ranges_valid=True,lengths_valid=True),peak_gpu_memory_mb=peak,
            artifacts=dict(probe_seconds=probe,probe_batches=46,candidate_evaluations=candidates,model_calls=calls,gradient_calls=grad,checkpoint='provided FSMT only',
                targets_sha256='767bb0e16b63b426bae8ebb48f8cda40711566367322b60e1c1e01ad63eb8162'),
            limitations=['The target outputs are explicitly supplied by the task; this is input optimization.',
                'Reference already reaches 100; compare cost and robustness at matched score.',
                'Preserve provided token IDs; an unrelated BPE tokenizer would invalidate the model.',
                'Single walltime observation with warm-up/batch-order effects; not a repeated benchmark.',
                'model_calls is the instrumented search counter; the 46 probe batches, example/generation checks and official evaluator are outside this counter.'])
        if number==8:
            report['observed_errors']=[dict(row_id=21,exact=False,loss=.011319739744067192,generated_text='Mr President, there is no religious liberation in Vietnam.',source_length=13)]
            lesson='R08 chỉ exact23/24 và official97.2778; Kaggle public94.5555/private100. Gradient fallback cải thiện so top8 cũ22/24 nhưng chưa giữ100. Code15.16s còn chậm hơn R06 nên không chọn làm bản tiết kiệm. Row21 còn margin0.01132; first-order CE proposal và shortlist hữu hạn không bảo đảm tìm được nghiệm exact.'
        else:
            lesson='R09 exact24/24, official100 và Kaggle100/100. Candidate evaluations2600 so7832 của R06 (giảm66.8%); measured code8.2827s so11.9711s (giảm30.8% ở hai lần đo). Search counter85 calls gồm16 gradient calls. Gradient sinh shortlist trong ngữ cảnh hiện tại rồi exact-verify margin; code vẫn có top8 fallback. Không có counter số lần fallback nên không quy mọi gain cho gradient riêng. Thời gian chịu warm-up/khác batch, chưa phải benchmark nhiều seed.'
        notes=f"# Inverse Translation R{number:02d}\n\n{lesson}\n\nCanary bốn provided examples khớp ở mỗi run (8/8 checks trong batch), BATCH_RESULT có đủR08/R09. Peak {peak:.2f}MiB trên T4. Không sửa frozen model weights.\n\n[5 paper và thiết kế](../../../research_v2/inverse-translation.md)."
        record_observation(report,URL,5,374816257,notes)
        scores.append(dict(run=f'R{number:02d}',public=public,private=100.,version_id=374816257,output_file=f'inverse-translation/R{number:02d}/submission.csv',
            description_note=f'Post-contest practice inverse-translation R{number:02d}: paper-guided frozen experiment; source/version/path verified; Complete (after deadline).'))
    record('inverse-translation',scores,'Competition score','maximize','https://www.kaggle.com/competitions/inverse-translation-aicc-round-10/submissions')
    q=json.loads((BASE/'queue.json').read_text());t=next(t for t in q['items'] if t['id']=='inverse-translation')
    old=t.pop('pending_research_batch',{})
    t['completed_research_batch']={**old,'observed_status':'Completed','runs':['R08','R09'],'followup':'Both predictions submitted and scored; reports recorded. No duplicate job started.'}
    t['status']='completed'
    (BASE/'queue.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
    render_status()
    chosen=json.loads((BASE/'BEST_CONFIGS.json').read_text());r=json.loads((BASE/'inverse-translation/runs/R09/report.json').read_text())
    chosen['inverse-translation']=dict(run_id='R09',selection_criterion='official_score100_exact24of24_with_fewer_measured_candidate_evaluations_than_R06; runtime_single_observation',
        validation=r['validation'],public_score=100.,private_score=100.,source_commit=COMMIT,notebook_url=URL,notebook_version=5,script_version_id=374816257,output_file='inverse-translation/R09/submission.csv')
    (BASE/'BEST_CONFIGS.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2)+'\n')
    fields=['task','run','metric','value','public_score','private_score','code_seconds','device','source_commit','notebook_version','script_version_id']
    with (BASE/'EXPERIMENTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for task in q['items']:
            for row in task['runs']:
                if row['status']!='completed':continue
                p=BASE/task['id']/'runs'/row['run_id'];r=json.loads((p/'report.json').read_text());e=json.loads((p/'evidence.json').read_text())
                writer.writerow(dict(task=task['id'],run=r['run_id'],metric=r['validation']['metric'],value=r['validation']['value'],public_score=r.get('leaderboard_public'),private_score=r.get('leaderboard_private'),
                    code_seconds=r['elapsed_seconds'],device=r['device'],source_commit=r['source_commit'],notebook_version=e['version'],script_version_id=e['script_version_id']))
    checked=audit();(BASE/'AUDIT.json').write_text(json.dumps(checked,indent=2)+'\n');assert checked['success'] and checked['successful_unique_experiments']==51 and checked['scored_unique_submissions']==51
    print(json.dumps(checked,indent=2))
if __name__=='__main__':main()
