"""Word Lookups: scratch BPE language models and offset-safe BMES decoding."""
from collections import Counter
import unicodedata

WORD_CONFIGS={
 1:dict(idea="BPE causal Transformer surprisal mapped back to character offsets",reverse=False,normalize=False,cohesion=False,dp=False,punct=False),
 2:dict(idea="Bidirectional surprisal sees boundaries from both contexts",reverse=True,normalize=False,cohesion=False,dp=False,punct=False),
 3:dict(idea="Sentence-level normalization tests surprisal scale shift",reverse=False,normalize=True,cohesion=False,dp=False,punct=False),
 4:dict(idea="Train-only character cohesion supplements BPE-internal boundaries",reverse=False,normalize=False,cohesion=True,dp=False,punct=False),
 5:dict(idea="Dynamic programming adds a word-length prior",reverse=False,normalize=False,cohesion=False,dp=True,punct=False),
 6:dict(idea="Explicit punctuation boundaries enforce consistent BMES",reverse=False,normalize=False,cohesion=False,dp=False,punct=True),
 7:dict(idea="Combine bidirectional surprise, cohesion, DP and punctuation",reverse=True,normalize=True,cohesion=True,dp=True,punct=True)
}
WORD_CONFIGS[8]={**WORD_CONFIGS[7],"idea":"Accessor-variety lexical spans supplement boundary surprise","accessor":True}
WORD_CONFIGS[9]={**WORD_CONFIGS[8],"idea":"Accessor spans plus BPE-dropout regularized scratch LM","bpe_dropout":.1}


class BoundaryLanguageModel(nn.Module):
    def __init__(self,vocab,dim=96):
        super().__init__()
        self.embedding=nn.Embedding(vocab,dim,padding_idx=0)
        self.position=nn.Embedding(128,dim)
        block=nn.TransformerEncoderLayer(dim,4,dim*3,dropout=0.1,batch_first=True,
                                         activation="gelu",norm_first=True)
        self.encoder=nn.TransformerEncoder(block,2,enable_nested_tensor=False)
        self.head=nn.Linear(dim,vocab)
    def forward(self,ids):
        x=self.embedding(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        causal=torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).triu(1)
        x=self.encoder(x,mask=causal,src_key_padding_mask=ids.eq(0))
        return self.head(x)


def as_chars(value):
    if isinstance(value,str) and value.lstrip().startswith("["):
        value=ast.literal_eval(value)
    return list(value) if not isinstance(value,str) else list(value)


def tags_from_boundaries(n,boundaries):
    tags=[];start=0
    for end in sorted(set(boundaries)|{n}):
        size=end-start
        if size<=0:continue
        tags.extend(["S"] if size==1 else ["B"]+["M"]*(size-2)+["E"])
        start=end
    assert len(tags)==n
    return tags


def run_word(run_id):
    cfg={**WORD_CONFIGS[run_id],"bpe_vocab":8192,"epochs":8,"budget_seconds":240,"max_length":96}
    info=start_run("word-lookups",run_id,cfg)
    root=find_data("aicc-round-9-word-lookups")
    train=pd.read_csv(root/"train.csv");test=pd.read_csv(root/"test.csv")
    train_text=["".join(as_chars(v)) for v in train.chars]
    test_chars=[as_chars(v) for v in test.chars];test_text=["".join(c) for c in test_chars]
    indices=np.random.default_rng(SEED).permutation(len(train_text))
    cut=max(1,int(len(indices)*0.9));tr_idx,va_idx=indices[:cut],indices[cut:]
    corpus=[train_text[i] for i in tr_idx]
    fit_text=corpus+[s[::-1] for s in corpus] if cfg["reverse"] else corpus
    tokenizer=make_bpe(fit_text,cfg["bpe_vocab"])
    tokenizer.add_special_tokens(["[FWD]","[BWD]"])
    fwd_id,bwd_id=tokenizer.token_to_id("[FWD]"),tokenizer.token_to_id("[BWD]")
    tokenizer.save(str(OUT/"tokenizer.json"))
    def lm_ids(texts,reverse=False):
        x=np.zeros((len(texts),cfg["max_length"]),dtype=np.int64)
        for i,text in enumerate(texts):
            ids=[bwd_id if reverse else fwd_id]+tokenizer.encode(text[::-1] if reverse else text).ids+[3]
            ids=ids[:cfg["max_length"]]
            x[i,:len(ids)]=ids
        return x
    tr=lm_ids(corpus)
    if cfg["reverse"]:tr=np.concatenate([tr,lm_ids(corpus,True)],0)
    va=lm_ids([train_text[i] for i in va_idx])
    loader=DataLoader(TensorDataset(torch.from_numpy(tr)),batch_size=96,shuffle=True)
    model=BoundaryLanguageModel(tokenizer.get_vocab_size()).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=0.01)
    scaler=torch.amp.GradScaler("cuda")
    history=[];best=float("inf");best_state=None;tick=time.time()
    def lm_validation():
        model.eval();total=0.;count=0
        with torch.inference_mode():
            for (x,) in DataLoader(TensorDataset(torch.from_numpy(va)),batch_size=128):
                x=x.to(DEVICE)
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=model(x[:,:-1])
                    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),ignore_index=0,reduction="sum")
                total+=float(loss);count+=int(x[:,1:].ne(0).sum())
        return total/max(count,1)
    for epoch in range(cfg["epochs"]):
        if cfg.get('bpe_dropout'):
            tokenizer.model.dropout=cfg['bpe_dropout']
            try:
                tr=lm_ids(corpus)
                if cfg['reverse']:tr=np.concatenate([tr,lm_ids(corpus,True)],0)
            finally:tokenizer.model.dropout=None
            loader=DataLoader(TensorDataset(torch.from_numpy(tr)),batch_size=96,shuffle=True)
        model.train();losses=[]
        for (x,) in loader:
            x=x.to(DEVICE);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(x[:,:-1])
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),ignore_index=0)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()))
            if time.time()-tick>cfg["budget_seconds"]:break
        nll=lm_validation();record=dict(epoch=epoch+1,train_nll=float(np.mean(losses)),validation_nll=nll)
        history.append(record);print("EPOCH "+json.dumps(record),flush=True)
        if nll<best:
            best=nll;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if time.time()-tick>cfg["budget_seconds"]:break
    model.load_state_dict(best_state);model.eval()
    unigram=Counter(c for s in corpus for c in s)
    bigram=Counter(pair for s in corpus for pair in zip(s,s[1:]))
    corpus_count=sum(unigram.values())
    bpe_lengths=[end-start for s in corpus[:4000] for start,end in tokenizer.encode(s).offsets if end>start]
    mean_length=float(np.clip(np.mean(bpe_lengths),1.4,3.0))
    lexical_bonus={}
    if cfg.get('accessor'):
        frequency=Counter(s[i:i+k] for s in corpus for k in [2,3,4] for i in range(max(0,len(s)-k+1)))
        retained={s for s,c in frequency.items() if c>=5}
        left={s:set() for s in retained};right={s:set() for s in retained}
        starts=Counter();ends=Counter()
        for s in corpus:
            for k in [2,3,4]:
                for i in range(max(0,len(s)-k+1)):
                    word=s[i:i+k]
                    if word not in retained:continue
                    if i:left[word].add(s[i-1])
                    else:starts[word]+=1
                    if i+k<len(s):right[word].add(s[i+k])
                    else:ends[word]+=1
        for word in retained:
            av=min(len(left[word])+starts[word],len(right[word])+ends[word])
            if av>=2:lexical_bonus[word]=min(.5,.2*math.log1p(av))
        del frequency,left,right,retained
        print('ACCESSOR_LEXICON '+json.dumps(dict(entries=len(lexical_bonus),min_frequency=5,max_span=4,weight=.2,cap=.5)),flush=True)
    @torch.inference_mode()
    def token_surprise(text,reverse=False):
        raw=text[::-1] if reverse else text
        encoded=tokenizer.encode(raw)
        values=np.zeros(max(0,len(text)-1),dtype=np.float32)
        seen=np.zeros(len(values),dtype=bool)
        # Sliding chunks avoid dropping the tail of long documents.
        tokens=encoded.ids;offsets=encoded.offsets
        stride=cfg["max_length"]-2
        for begin in range(0,len(tokens),stride):
            end=min(len(tokens),begin+stride)
            sequence=[bwd_id if reverse else fwd_id]+tokens[begin:end]
            x=torch.tensor(sequence,device=DEVICE)[None]
            with torch.autocast("cuda",dtype=torch.float16):logits=model(x).float()
            for j in range(begin+1,end):
                pos=offsets[j][0]
                if reverse:pos=len(text)-pos
                if 1<=pos<len(text):
                    local=j-begin
                    score=-float(logits[0,local].log_softmax(-1)[tokens[j]])
                    values[pos-1]=score;seen[pos-1]=True
        if seen.any():
            fallback=float(values[seen].quantile(0.25)) if hasattr(values[seen],"quantile") else float(np.quantile(values[seen],0.25))
            values[~seen]=fallback-1
        return values
    def boundary_score(text):
        score=token_surprise(text)
        if cfg["reverse"]:score=(score+token_surprise(text,True))/2
        if len(score):
            if cfg["normalize"]:score=(score-score.mean())/max(score.std(),0.1)
            if cfg["cohesion"]:
                pmi=np.array([math.log((bigram[a,b]+0.5)*corpus_count/((unigram[a]+1)*(unigram[b]+1))) for a,b in zip(text,text[1:])])
                pmi=(pmi-pmi.mean())/max(pmi.std(),0.1)
                score=(score-score.mean())/max(score.std(),0.1)-0.6*pmi
        return score
    sampled=np.concatenate([boundary_score(s) for s in corpus[:256] if len(s)>1])
    threshold=float(np.quantile(sampled,1-1/mean_length))
    def decode(text):
        n=len(text)
        if n<2:return ["S"]*n
        score=boundary_score(text)
        forced={i for i in range(1,n) if unicodedata.category(text[i-1]).startswith("P") or unicodedata.category(text[i]).startswith("P")} if cfg["punct"] else set()
        if not cfg["dp"]:boundaries=set(np.flatnonzero(score>=threshold)+1)
        else:
            probability=1/(1+np.exp(-np.clip((score-threshold)/max(sampled.std(),0.1),-8,8)))
            dp=np.full(n+1,-np.inf);prev=np.zeros(n+1,dtype=int);dp[0]=0
            for end in range(1,n+1):
                for start in range(max(0,end-6),end):
                    if any(start<b<end for b in forced):continue
                    length=end-start
                    interior=float(np.log1p(-probability[start:end-1]).sum())
                    boundary=float(math.log(probability[end-1])) if end<n else 0.
                    value=dp[start]+interior+boundary-0.08*(length-mean_length)**2+lexical_bonus.get(text[start:end],0.)
                    if value>dp[end]:dp[end]=value;prev[end]=start
            boundaries=set();end=n
            while end>0:
                if end<n:boundaries.add(end)
                end=int(prev[end])
        boundaries|=forced
        return tags_from_boundaries(n,boundaries)
    predictions=[]
    for i,text in enumerate(test_text):
        tags=decode(text);assert len(tags)==len(test_chars[i])
        predictions.append(str(tags))
        if i%1000==0:print("DECODE "+str(i),flush=True)
    sub=pd.DataFrame(dict(id=test.id,bio_tags=predictions))
    checks=dict(ids_unique=sub.id.is_unique,schema_valid=list(sub.columns)==["id","bio_tags"],
                row_count=len(sub),lengths_match=True,valid_bmes=True)
    torch.save(best_state,OUT/"model.pt")
    return finish_run(info,dict(metric="BPE_LM_NLL_proxy",value=best,direction="minimize",
          samples=len(va_idx),split="train_only_10pct",boundary_f1=None),sub,checks,history,
          artifacts=dict(tokenizer="tokenizer.json",pretrained_weights=False,train_only_statistics=True,
                         threshold=threshold,mean_bpe_length=mean_length,accessor_lexicon_entries=len(lexical_bonus)),
          limitations=["LM NLL is a proxy, not segmentation F1; no segmentation gold labels were used.",
                       "BPE boundaries and linguistic words differ; decoding is a testable hypothesis.",
                       "Only training text is used for vocabulary, LM fitting and threshold calibration.",
                       "Accessor bonus is a transfer hypothesis, not a full segmental neural LM or Bayesian word model.",
                       "BPE-dropout uses tokenizers stochastic merges; seed42 alone does not guarantee identical merge samples across library versions."])

