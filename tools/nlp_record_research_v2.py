"""Record compact fields observed in rendered Kaggle outputs, not replayed metrics."""
import ast,json
from pathlib import Path
from nlp_record_observation import record_observation
from nlp_record_scores import record
BASE=Path(__file__).resolve().parents[1]/'experiments/nlp_aicc'
FIRST='342c70da46ccb69e3a99f46313088efbcb0d09db'
SECOND='84e9a9efaa6346578162fa12915a8368f18e6485'
SPECS=[
 ('lost-interpreter',9,15.813640594482422,102.09048461914062,47.61962890625,14.9744,17.8852),
 ('lost-interpreter',10,16.317798614501953,84.42564630508423,47.498046875,15.2480,18.4884),
 ('alchemy',10,5/30,13.601986408233643,26.595703125,.0285,.1142),
 ('alchemy',11,1/30,.34708428382873535,17.84716796875,0.,0.),
 ('moment-match',11,.6572237960339944,65.00148248672485,230.0283203125,.6304,.6233),
 ('moment-match',12,.6940509915014165,86.56071758270264,211.8720703125,.6456,.6253),
 ('word-lookups',8,5.802002440784768,230.56555938720703,980.083984375,.9150,.9171),
 ('word-lookups',9,5.789452881850572,244.07154321670532,981.7841796875,.9145,.9168),
]
META={
 'lost-interpreter':('lost_interpreter','LOST_CONFIGS',374731593,8,3000,5000,'MAE','minimize','fixed_train_holdout_15pct'),
 'alchemy':('alchemy','ALCHEMY_CONFIGS',374731690,5,30,70,'bijective_holdout_accuracy','maximize','fixed_result_disjoint_train_holdout'),
 'moment-match':('moment_match','MOMENT_CONFIGS',374734057,11,1059,1970,'accuracy','maximize','provided_validation'),
 'word-lookups':('word_lookups','WORD_CONFIGS',374733892,3,3709,9272,'BPE_LM_NLL_proxy','minimize','train_only_10pct'),
}
LEARNED={
 ('lost-interpreter',9):'Typed input giảm lỗi ≤4 lệnh (5.91→5.17), nhưng tăng lỗi5–8/9–16 (17.53/23.00) và MAE tổng15.81 > R08 15.31. Public/private cũng kém R08. Chưa thay thế BPE BiGRU. Mượn typed inputs từ NPI, không có trace để reproduce NPI.',
 ('lost-interpreter',10):'Curriculum hỗn hợp không cải thiện: val16.3178, public15.2480/private18.4884. So R09 là ablation curriculum cùng architecture; không kết luận curriculum luôn vô dụng. Lỗi chương trình dài vẫn chi phối.',
 ('alchemy',10):'Matching vuông đúng giả định, finite canary và tổng hàng/cột đã qua; val vẫn5/30, public/private giống R09 .0285/.1142. Sửa objective chưa thay được tri thức nghĩa. 24/30 val results chưa thấy dưới bất kỳ vai trò train nào; 0 UNK nên không đổ lỗi cho tokenization.',
 ('alchemy',11):'Lexical zero-shot ridge thất bại: val1/30 và Kaggle0/0. Chia sẻ character/BPE không cung cấp semantic attributes như ESZSL gốc. Không xem string similarity là tri thức nghĩa; không tự thêm pretrained weights.',
 ('moment-match',11):'Cross-attention event-level chưa tăng: val.6572, Kaggle.6304/.6233. Event pooling trước align có thể mất tín hiệu local token; attention thêm không tự tạo reasoning tốt hơn. Cần kiểm lỗi, không tăng depth mù.',
 ('moment-match',12):'Soft-DTW tăng val từ.691218 lên.694051 (3/1059 câu), public.6456/private.6253. Chênh nhỏ và private không tăng so R04; chưa tuyên bố cải thiện ổn định. Đây là full soft-min path, không phải R08 strict subsequence hoặc Drop-DTW.',
 ('word-lookups',8):'Accessor-variety span bonus tăng Boundary F1 public.9097→.9150, private.9113→.9171; LM NLL giữ nguyên5.8020. 67397 span train-only, không dùng segmentation labels. Đây là cải thiện decoding ở cùng LM recipe, không phải một LM tốt hơn.',
 ('word-lookups',9):'BPE-dropout NLL5.78945 thấp hơn R08 nhưng Boundary F1.9145/.9168 thấp hơn.9150/.9171. NLL không thay được metric cuối. Giữ R08 làm preset tốt hơn trong lượt này; không tối ưu dropout tiếp theo private score.',
}
def config_table(module,table):
    parsed=ast.parse((BASE/'code'/f'{module}.py').read_text())
    # Only literal configuration assignments; no import, model, data or training execution.
    assignments=[n for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==table or isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id==table for t in n.targets)]
    ns={};exec(compile(ast.Module(body=assignments,type_ignores=[]),'config-only','exec'),ns)
    return ns[table]
def main():
    scores={}
    for task,number,value,elapsed,peak,public,private in SPECS:
        module,table,version_id,version,samples,rows,metric,direction,split=META[task]
        cfg=dict(config_table(module,table)[number])
        if task=='lost-interpreter':cfg.update(epochs=60,budget_seconds=300,batch_size=128,bpe_vocab=2048,pretrained=False)
        if task=='alchemy' and number==10:cfg.update(steps=240,budget_seconds=180,bpe_vocab=1024,pretrained=False)
        if task=='alchemy' and number==11:cfg.update(pretrained=False,bpe_vocab=1024,attribute_dim=64)
        if task=='moment-match':cfg.update(epochs=30,budget_seconds=240,batch_size=48)
        if task=='word-lookups':cfg.update(bpe_vocab=8192,epochs=8,budget_seconds=240,max_length=96)
        schema={'lost-interpreter':['ID','output'],'alchemy':['Id','result'],'moment-match':['row_id','label'],'word-lookups':['id','bio_tags']}[task]
        artifact=dict(pretrained_weights=False)
        if task=='lost-interpreter':
            means=[5.1723341942,17.5331497192,23.0000934601] if number==9 else [5.3949670792,18.2506790161,23.3719425201]
            artifact.update(tokenizer='train-only BPE',max_instructions=10,max_operands=5,max_tokens_per_instruction=11,
                error_by_length=dict(bins=['(0,4]','(4,8]','(8,16]'],mean=means,count=[757,1475,768]))
        if task=='alchemy':artifact.update(tokenizer='train-only BPE' if number==10 else 'train-only BPE plus character attributes',training_rows=120,
            coverage_diagnostics=dict(unique_train_names=206,heldout_results=dict(count=30,seen_any=6,seen_as_input=6,unk_tokens=0),test_candidates=dict(count=70,seen_any=17,seen_as_input=17,unk_tokens=0),test_input_names=dict(count=140,seen_any=88,seen_as_input=79,unk_tokens=0)))
        if task=='word-lookups':artifact.update(tokenizer='tokenizer.json',train_only_statistics=True,accessor_lexicon_entries=67397,mean_bpe_length=1.4,threshold=-1.0258628704190802 if number==8 else -1.019620548518612)
        if task=='moment-match':artifact.update(weights='provided_fastText_only',word_feature_view=False);artifact.pop('pretrained_weights')
        report=dict(task=task,run_id=f'R{number:02d}',config=cfg,seed=42,device='Tesla T4',gpu_count=2,source_commit=FIRST if task in ['lost-interpreter','alchemy'] else SECOND,
            versions=dict(torch='2.11.0+cu130',numpy='2.1.3',pandas='2.3.3',**{'scikit-learn':'1.6.1','tokenizers':'0.23.3'}),status='completed',elapsed_seconds=elapsed,peak_gpu_memory_mb=peak,
            validation=dict(metric=metric,value=value,direction=direction,samples=samples,split=split),submission_checks=dict(ids_unique=True,schema_valid=True,row_count=rows),
            artifacts=artifact,limitations=['Compact fields observed in rendered output; full per-epoch history remains on Kaggle.','Single seed, no statistical significance claim.'])
        url=f'https://www.kaggle.com/code/shareaccc3a/aicc-nlp-{task}-learning-experiments?scriptVersionId={version_id}'
        notes=f"# {task} R{number:02d}\n\n{LEARNED[task,number]}\n\n[Paper và thiết kế](../../../research_v2/{task}.md). Code {elapsed:.3f}s, peak {peak:.2f}MiB trên T4."
        record_observation(report,url,version,version_id,notes)
        scores.setdefault(task,[]).append(dict(run=f'R{number:02d}',public=public,private=private,version_id=version_id,output_file=f'{task}/R{number:02d}/submission.csv',description_note='Paper-guided preset frozen before scoring; source version and path checked in submission form.'))
    for task,rows in scores.items():
        metric='Boundary F1' if task=='word-lookups' else 'MAE' if task=='lost-interpreter' else 'Accuracy'
        slug={'lost-interpreter':'lost-interpreter-aicc-round-5','alchemy':'alchemy-aicc-round-4','moment-match':'moment-match-aicc-round-10','word-lookups':'aicc-round-9-word-lookups'}[task]
        record(task,rows,metric,'minimize' if task=='lost-interpreter' else 'maximize',f'https://www.kaggle.com/competitions/{slug}/submissions')
    from nlp_status import render_status
    render_status();print('Recorded 8 observed research configurations and their actual Kaggle scores.')
if __name__=='__main__':main()
