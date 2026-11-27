"""Build self-contained study notebooks. Execution outputs are deliberately empty."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/nlp_aicc'
COMMIT='84e9a9efaa6346578162fa12915a8368f18e6485'
TASKS={
 'moment-match':dict(module='moment_match',entry='run_moment',run=12,title='Moment Match: late interaction → alignment prior',
   flow='Event text → train-only BPE → supplied fastText initialization → IDF-weighted token matching → event interaction grid → CNN score + Soft-DTW residual → choose one of four timelines.',
   result='R12 val69.4051%, public.6456/private.6253. Val R04=69.1218%, private.6284. Chênh3/1059 câu val chưa chứng minh tốt hơn ổn định.',
   learn='Một vector pooled sớm có thể bỏ mất local match. Cross-attention event-level R11 chưa giúp; Soft-DTW giữ trật tự nhưng vẫn ép match event nhiễu. So R04/R08/R11/R12 để thấy sự khác nhau.'),
 'lost-interpreter':dict(module='lost_interpreter',entry='run_lost',run=8,title='Lost Interpreter: recurrence, numbers and execution',
   flow='Program string → train-only BPE → embedding128 → packed two-layer BiGRU → masked mean/max → scalar head → SmoothL1 on standardized output → integer prediction.',
   result='R08 val MAE15.3089, public14.1608/private16.7920. Typed instruction R09=15.8136, mixed curriculum R10=16.3178 trên cùng holdout.',
   learn='BiGRU nhỏ vượt Transformer trong điều kiện đã thử. Tách opcode/operand chưa đủ học đúng state transitions. Đổi RUN_ID sang9/10 để học typed representation và curriculum; không nhầm với NPI có trace supervision.'),
 'word-lookups':dict(module='word_lookups',entry='run_word',run=8,title='Word Lookups: language model → word boundaries',
   flow='Unsegmented characters → scratch BPE causal LM trained in both directions → surprisal at character offsets + PMI → train-only threshold → DP with punctuation, length prior and accessor-variety spans → BMES.',
   result='R08 Boundary F1 public.9150/private.9171, từR07 .9097/.9113. R09 BPE-dropout NLL giảm nhưng F1.9145/.9168 thấp hơn.',
   learn='NLL không phải segmentation F1. Một cụm xuất hiện trong nhiều ngữ cảnh trái/phải là tín hiệu lexical bổ sung. RUN_ID7/8 cô lập span bonus; RUN_ID8/9 kiểm tra dropout. Không dùng test gold.'),
 'inverse-translation':dict(module='inverse_translation',entry='run_inverse',run=6,title='Inverse Translation: optimize discrete inputs',
   flow='Provided FSMT + target token IDs → verify four provided examples → stream vocabulary probes → coordinate substitutions minimizing target decision margin → verify actual generated tokens → official evaluator.',
   result='R06 official100 exact24/24; Kaggle100/100; code11.971s,7832 candidate evaluations. Gradient R08/R09 có code nghiên cứu, kết quả xem STATUS mới nhất.',
   learn='Giữ token IDs của model, không thay BPE khác. Model frozen là một phần bắt buộc của đề. Chỉ gọi tiết kiệm khi giữ exact score; tham khảo HotFlip và AutoPrompt cho RUN_ID8/9.'),
 'alchemy':dict(module='alchemy',entry='run_alchemy',run=10,title='Alchemy: commutativity and global assignment',
   flow='Item names → scratch BPE → shared lexical roles → symmetric additive/residual composition → candidate similarity matrix → balanced square matching loss → Hungarian bijection.',
   result='R10 val5/30, Kaggle.0285/.1142, bằngR09. R11 lexical bilinear compatibility1/30, Kaggle0/0. Đây là mô hình để học matching, chưa phải lời giải mạnh.',
   learn='Ràng buộc đúng không tạo ra nghĩa của từ chưa thấy. 24/30 val results chưa xuất hiện ởtrain, dù0 UNK. Không tải BERT pretrained. RUN_ID9/10 so old loss và square matching; R11 kiểm tra giới hạn lexical attributes.'),
}
def cell(kind,text):
    c=dict(cell_type=kind,metadata={},source=text.splitlines(True))
    if kind=='code':c.update(execution_count=None,outputs=[])
    return c
def generate():
    out=BASE/'notebooks/research_v2';out.mkdir(parents=True,exist_ok=True)
    common=(BASE/'code/common.py').read_text();manifest={}
    for task,cfg in TASKS.items():
        module=(BASE/'code'/f"{cfg['module']}.py").read_text()
        intro=f"# {cfg['title']}\n\n{cfg['flow']}\n\n**Điểm đã đo ở các version Kaggle trước:** {cfg['result']}\n\n**Bài học:** {cfg['learn']}\n\nNotebook này chứa source đầy đủ nhưng chưa execute tại bản export này; execution_count=null và outputs rỗng là có chủ ý. Không suy diễn metric mới từ notebook chưa chạy.\n\nBật GPU T4 trên Kaggle, attach competition tương ứng trước Run All. Loader tự tìm cả `/kaggle/input/competitions/<slug>` và `/kaggle/input/<slug>`. Code assert T4, không tự chuyển CPU. Thực hành sau contest, không dùng trong thời gian cuộc thi cấm trợ giúp.\n"
        paper=(BASE/'research_v2'/f'{task}.md').read_text()
        setup=f"SOURCE_COMMIT = {COMMIT!r}\nRUN_ID = {cfg['run']}\n"
        if task=='inverse-translation':
            setup+='import subprocess, sys\nsubprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.57.1"],check=True)\n'
        final=f"OUT = Path('/kaggle/working') / {task!r} / f'R{{RUN_ID:02d}}'\nOUT.mkdir(parents=True, exist_ok=True)\nreport = {cfg['entry']}(RUN_ID)\nprint('Submission:', OUT / 'submission.csv')\n"
        cells=[cell('markdown',intro),cell('markdown',paper),cell('code',setup),
               cell('markdown','## Shared utilities\n\nSeed, BPE, masking, data paths, schema checks và report writer.\n'),cell('code',common),
               cell('markdown','## Model và experiments\n\nCác config Rxx nằm đầu source. Đổi RUN_ID trước khi Run All; mỗi lượt phải bắt đầu model/optimizer mới.\n'),cell('code',module),cell('code',final)]
        for c in cells:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        nb=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
             language_info=dict(name='python'),nlp_provenance=dict(source_commit=COMMIT,export_executed=False,task=task)),cells=cells)
        p=out/f'{task}_study.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        manifest[task]=dict(path=str(p.relative_to(ROOT)),source_commit=COMMIT,run_id=cfg['run'],executed=False,
            common_sha256=hashlib.sha256(common.encode()).hexdigest(),module_sha256=hashlib.sha256(module.encode()).hexdigest())
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
if __name__=='__main__':
    m=generate();print(json.dumps(dict(notebooks=len(m),all_code_cells_parse=True,export_executed=False)))
