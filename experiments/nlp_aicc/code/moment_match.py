"""Moment Match: extend official late interaction with BPE and contextual alignment."""
# Reference: AI-Community-Contest/solutions/round-10/moment-match.ipynb (MIT).
from collections import Counter


MOMENT_CONFIGS = {
 1: dict(idea="BPE late interaction plus soft alignment", vocab=2048, token_layers=0, event_context=False, soft=True, temperature=0.2, idf=False, center=False),
 2: dict(idea="Contextualize BPE tokens before late interaction", vocab=2048, token_layers=1, event_context=False, soft=True, temperature=0.2, idf=False, center=False),
 3: dict(idea="Contextualize the event sequence with positional attention", vocab=2048, token_layers=0, event_context=True, soft=True, temperature=0.2, idf=False, center=False),
 4: dict(idea="IDF weighted token matching suppresses common words", vocab=2048, token_layers=0, event_context=False, soft=True, temperature=0.2, idf=True, center=False),
 5: dict(idea="Sharper soft matching tests alignment entropy", vocab=2048, token_layers=0, event_context=False, soft=True, temperature=0.07, idf=False, center=False),
 6: dict(idea="Center-only ablation measures the contribution of neighboring events", vocab=2048, token_layers=0, event_context=False, soft=True, temperature=0.2, idf=False, center=True),
 7: dict(idea="Combine token contextualization and IDF after R03/R04 evidence", vocab=2048, token_layers=1, event_context=False, soft=True, temperature=0.2, idf=True, center=False)
}


# Hypotheses added after the first seven measured experiments.
MOMENT_CONFIGS[8]={**MOMENT_CONFIGS[4],"idea":"IDF plus a differentiable monotonic subsequence alignment prior","monotonic":True}
MOMENT_CONFIGS[9]={**MOMENT_CONFIGS[4],"idea":"Hard max word-piece alignment instead of soft matching","hard":True}


def monotonic_alignment(scores,gamma=0.15):
    """Mean score, softly averaged over strictly increasing subsequence paths.

    Not the complete Soft-DTW algorithm: no repeats or horizontal moves. This
    small task-specific prior is inspired by differentiable dynamic programming.
    """
    from math import comb
    n,m=scores.shape[-2:]
    if n>m:raise ValueError("A strict subsequence needs n <= m")
    dp=scores[...,0,:]
    for i in range(1,n):
        values=[]
        for j in range(m):
            if j<i:values.append(scores.new_full(scores.shape[:-2],-1e4))
            else:values.append(scores[...,i,j]+gamma*torch.logsumexp(dp[..., :j]/gamma,dim=-1))
        dp=torch.stack(values,dim=-1)
    log_partition=gamma*torch.logsumexp(dp[...,n-1:]/gamma,dim=-1)
    return (log_partition-gamma*math.log(comb(m,n)))/n


def alignment_canary():
    from itertools import combinations
    toy=(torch.arange(15,device=DEVICE,dtype=torch.float32).reshape(1,1,3,5)%7)/7
    toy.requires_grad_(True);gamma=.15
    got=monotonic_alignment(toy,gamma)
    paths=torch.stack([sum(toy[...,i,j] for i,j in enumerate(path)) for path in combinations(range(5),3)],-1)
    expected=(gamma*torch.logsumexp(paths/gamma,-1)-gamma*math.log(paths.shape[-1]))/3
    assert torch.allclose(got,expected,atol=1e-6)
    got.sum().backward();assert torch.isfinite(toy.grad).all() and (toy.grad>=0).all()
    print("DP_CANARY_PASS "+json.dumps(dict(enumerated_paths=10,value=float(got.detach().cpu().item()))),flush=True)


class MomentMatcher(nn.Module):
    def __init__(self, initial, idf, config, dim=96):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding.from_pretrained(torch.as_tensor(initial), freeze=False, padding_idx=0)
        self.projection = nn.Linear(initial.shape[1], dim, bias=False)
        self.register_buffer("idf", torch.as_tensor(idf, dtype=torch.float32))
        if config["token_layers"]:
            block = nn.TransformerEncoderLayer(dim, 4, dim*2, dropout=0.1, batch_first=True,
                                               activation="gelu", norm_first=True)
            self.tokens = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
            self.token_position = nn.Embedding(24, dim)
        if config["event_context"]:
            self.event_position = nn.Embedding(9, dim)
            block = nn.TransformerEncoderLayer(dim, 4, dim*2, dropout=0.1, batch_first=True,
                                               activation="gelu", norm_first=True)
            self.events = nn.TransformerEncoder(block, 1, enable_nested_tensor=False)
            self.context_gate = nn.Parameter(torch.tensor(0.0))
        self.grid = nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.GELU(),
                                  nn.Conv2d(32,32,3,padding=1),nn.GELU())
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(32*5*9,96),nn.GELU(),
                                  nn.Dropout(0.2),nn.Linear(96,1))

        if config.get("monotonic",False):
            self.alignment_scale=nn.Parameter(torch.tensor(.2))

    def representations(self, ids):
        shape=ids.shape
        x=self.projection(self.emb(ids))
        if self.config["token_layers"]:
            z=x.reshape(-1,shape[-1],x.shape[-1])
            positions=self.token_position(torch.arange(shape[-1],device=ids.device))
            z=self.tokens(z+positions,src_key_padding_mask=ids.reshape(-1,shape[-1]).eq(0))
            x=z.reshape(x.shape)
        if self.config["event_context"]:
            mask=ids.ne(0).unsqueeze(-1)
            event=(x*mask).sum(-2)/mask.sum(-2).clamp(min=1)
            e=event.reshape(-1,event.shape[-2],event.shape[-1])
            e=self.events(e+self.event_position(torch.arange(e.shape[1],device=ids.device)))
            x=x+torch.sigmoid(self.context_gate)*e.reshape(event.shape).unsqueeze(-2)
        return F.normalize(x,dim=-1)

    def forward(self, targets,candidates):
        tm,cm=targets.ne(0),candidates.ne(0)
        tv,cv=self.representations(targets),self.representations(candidates)
        sim=torch.einsum("btwd,bcevd->bctewv",tv,cv).float()
        valid=tm[:,None,:,None,:,None]&cm[:,:,None,:,None,:]
        sim=sim.masked_fill(~valid,-1e4)
        temp=self.config["temperature"]
        if self.config.get("hard",False):
            left,right=sim.amax(-1),sim.amax(-2)
        else:
            left=(sim*(sim/temp).softmax(-1)).sum(-1)
            right=(sim*(sim/temp).softmax(-2)).sum(-2)
        tw=self.idf[targets] if self.config["idf"] else tm.float()
        cw=self.idf[candidates] if self.config["idf"] else cm.float()
        tw=tw*tm
        cw=cw*cm
        left=(left*tw[:,None,:,None,:]).sum(-1)/tw[:,None,:,None,:].sum(-1).clamp(min=1e-6)
        right=(right*cw[:,:,None,:,:]).sum(-1)/cw[:,:,None,:,:].sum(-1).clamp(min=1e-6)
        grids=torch.stack([left,right],dim=2)
        if self.config["center"]:
            keep=torch.zeros(5,device=targets.device);keep[2]=1
            grids=grids*keep[None,None,None,:,None]
        scores=self.head(self.grid(grids.flatten(0,1))).reshape(-1,4)
        if self.config.get("monotonic",False):
            alignment=monotonic_alignment((left+right)/2)
            scores=scores+self.alignment_scale*alignment
        return scores


def run_moment(run_id):
    config={**MOMENT_CONFIGS[run_id], "epochs":30, "budget_seconds":240, "batch_size":48}
    info=start_run("moment-match",run_id,config)
    if config.get("monotonic",False):alignment_canary()
    root=find_data("moment-match-aicc-round-10")
    texts={s:json.loads((root/f"texts_{s}.json").read_text()) for s in ["train","validation","test"]}
    labels={s:np.load(root/f"labels_{s}.npy") for s in ["train","validation"]}
    events=[line for target,cands in zip(texts["train"]["target"],texts["train"]["candidates"])
            for sequence in [target]+cands for line in sequence]
    tokenizer=make_bpe(events,config["vocab"])
    tokenizer.save(str(OUT/"tokenizer.json"))
    vocab=json.loads((root/"vocab.json").read_text())
    provided=np.load(root/"embedding_matrix.npy")
    initial=np.zeros((tokenizer.get_vocab_size(),provided.shape[1]),dtype=np.float32)
    counts=np.zeros(len(initial),dtype=np.float32)
    for word in sorted(set(w for event in events for w in event.split())):
        if word in vocab:
            for token in tokenizer.encode(word).ids:
                initial[token]+=provided[int(vocab[word])]
                counts[token]+=1
    initial/=np.maximum(counts[:,None],1)
    rng=np.random.default_rng(SEED)
    missing=(counts==0)
    initial[missing]=rng.normal(0,0.02,(int(missing.sum()),provided.shape[1]))
    initial[0]=0
    df=Counter(token for event in events for token in set(tokenizer.encode(event).ids))
    idf=np.array([math.log((len(events)+1)/(df[i]+1))+1 for i in range(len(initial))],dtype=np.float32)
    idf[0]=0
    encoded={}
    for split,t in texts.items():
        a=np.stack([encode_texts(tokenizer,seq,24) for seq in t["target"]])
        b=np.stack([np.stack([encode_texts(tokenizer,seq,24) for seq in cands]) for cands in t["candidates"]])
        encoded[split]=(torch.from_numpy(a),torch.from_numpy(b))
    train_data=TensorDataset(*encoded["train"],torch.as_tensor(labels["train"],dtype=torch.long))
    loader=DataLoader(train_data,batch_size=config["batch_size"],shuffle=True,num_workers=0)
    model=MomentMatcher(initial,idf,config).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    history=[];best=-1;best_state=None;bad=0;tick=time.time()
    def infer(split):
        model.eval();result=[]
        with torch.inference_mode():
            for a,b in DataLoader(TensorDataset(*encoded[split]),batch_size=48):
                with torch.autocast("cuda",dtype=torch.float16):
                    scores=model(a.to(DEVICE),b.to(DEVICE))
                assert torch.isfinite(scores).all()
                result.append(scores.float().cpu())
        return torch.cat(result)
    scaler=torch.amp.GradScaler("cuda")
    for epoch in range(config["epochs"]):
        model.train();losses=[]
        for a,b,y in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                scores=model(a.to(DEVICE),b.to(DEVICE));loss=F.cross_entropy(scores,y.to(DEVICE))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        scores=infer("validation")
        acc=float((scores.argmax(1).numpy()==labels["validation"]).mean())
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_accuracy=acc)
        history.append(row);print("EPOCH "+json.dumps(row),flush=True)
        if acc>best:
            best=acc;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=7 or time.time()-tick>config["budget_seconds"]:break
    model.load_state_dict(best_state)
    val_scores=infer("validation").numpy();test_scores=infer("test").numpy()
    assert np.isfinite(test_scores).all()
    np.save(OUT/"validation_logits.npy",val_scores)
    np.save(OUT/"validation_labels.npy",labels["validation"])
    probability=torch.softmax(torch.from_numpy(val_scores),1).numpy()
    predicted=val_scores.argmax(1);gold=labels["validation"]
    ordered=np.sort(probability,axis=1);margins=ordered[:,-1]-ordered[:,-2]
    wrong=np.flatnonzero(predicted!=gold)
    selected=wrong[np.argsort(-probability[wrong,predicted[wrong]])[:8]]
    cases=[dict(row_id=int(i),gold=int(gold[i]),prediction=int(predicted[i]),confidence=float(probability[i,predicted[i]]),
                target=texts["validation"]["target"][i],
                chosen=texts["validation"]["candidates"][i][predicted[i]],
                expected=texts["validation"]["candidates"][i][gold[i]]) for i in selected]
    (OUT/"validation_errors.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    print("VALIDATION_ERRORS "+json.dumps(cases,ensure_ascii=False),flush=True)
    print("VALIDATION_DIAGNOSTICS "+json.dumps(dict(mean_margin=float(margins.mean()),wrong_count=int(len(wrong)))),flush=True)
    torch.save(best_state,OUT/"model.pt")
    sub=pd.DataFrame(dict(row_id=np.arange(len(test_scores)),label=test_scores.argmax(1)))
    checks=dict(ids_unique=sub.row_id.is_unique,schema_valid=list(sub.columns)==["row_id","label"],
                row_count=len(sub),labels_valid=bool(sub.label.between(0,3).all()))
    return finish_run(info,dict(metric="accuracy",value=best,direction="maximize",
          samples=len(labels["validation"]),split="provided_validation"),sub,checks,history,
          artifacts=dict(tokenizer="tokenizer.json",weights="provided_fastText_only",
                         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() and Path(__file__).exists() else None),
          limitations=["BPE vocabulary trained on train text only; supplied fastText initializes its vectors.",
                       "Reference architecture is adapted, not claimed to reproduce its 0.7243 result."])

