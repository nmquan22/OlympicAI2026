"""Alchemy: scratch candidate-aware scoring with symmetry and global matching.
Architecture ideas adapted from the organizer's MIT-licensed BERT reference.
No pretrained weights or pretrained tokenizer are loaded.
"""
from scipy.optimize import linear_sum_assignment

ALCHEMY_CONFIGS={
 1:dict(idea="Scratch cross-encoder, swap consistency and Hungarian",hard=False,full=False,layers=2,probability=False,sinkhorn=False),
 2:dict(idea="Model-mined hard negatives with a random-negative mixture",hard=True,full=False,layers=2,probability=False,sinkhorn=False),
 3:dict(idea="CE over the complete training candidate list",hard=False,full=True,layers=2,probability=False,sinkhorn=False),
 4:dict(idea="Four scratch encoder layers test extra capacity",hard=False,full=False,layers=4,probability=False,sinkhorn=False),
 5:dict(idea="Average probabilities rather than logits of swapped inputs",hard=False,full=False,layers=2,probability=True,sinkhorn=False),
 6:dict(idea="Sinkhorn training models candidate competition before Hungarian",hard=False,full=False,layers=2,probability=False,sinkhorn=True),
 7:dict(idea="Combine hard negatives, full-list CE and probability averaging",hard=True,full=True,layers=2,probability=True,sinkhorn=False)
}


ALCHEMY_CONFIGS[8]={**ALCHEMY_CONFIGS[3],"idea":"Commutative three-tower composer with a shared scratch BPE embedding","composer":True}

ALCHEMY_CONFIGS[9]={**ALCHEMY_CONFIGS[8],"idea":"Tie input/result roles and use additive residual composition","tied_roles":True}

class AlchemyComposer(nn.Module):
    """Candidate embeddings share lexical features; item order is symmetric."""
    def __init__(self,vocab,dim=96,tied_roles=False):
        super().__init__()
        self.tied_roles=tied_roles
        self.embedding=nn.Embedding(vocab,dim,padding_idx=0)
        self.item=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.LayerNorm(dim))
        self.combine=nn.Sequential(nn.Linear(dim*3,dim*2),nn.GELU(),nn.Dropout(.1),nn.Linear(dim*2,dim))
        self.result=nn.Linear(dim,dim,bias=False)
        self.scale=nn.Parameter(torch.tensor(2.3))
    def forward(self,ids):
        segment=ids.eq(3).cumsum(1)
        valid=ids.gt(4)
        h=self.embedding(ids)
        vectors=[]
        for part in range(3):
            mask=(segment.eq(part)&valid).unsqueeze(-1)
            vectors.append((h*mask).sum(1)/mask.sum(1).clamp(min=1))
        a,b=self.item(vectors[0]),self.item(vectors[1])
        query=self.combine(torch.cat([a+b,(a-b).abs(),a*b],-1))
        if self.tied_roles:
            query=a+b+.1*query
            candidate=self.item(vectors[2])
        else:candidate=self.result(vectors[2])
        return (F.normalize(query,dim=-1)*F.normalize(candidate,dim=-1)).sum(-1)*self.scale.exp().clamp(max=30)


class AlchemyScorer(nn.Module):
    def __init__(self,vocab,layers):
        super().__init__()
        self.encoder=PositionEncoder(vocab,dim=96,layers=layers,max_length=64)
        self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x):
        h,mask=self.encoder(x)
        return self.head(pools(h,mask)).squeeze(-1)


def run_alchemy(run_id):
    cfg={**ALCHEMY_CONFIGS[run_id],"steps":240,"budget_seconds":180,"bpe_vocab":1024,"pretrained":False}
    info=start_run("alchemy",run_id,cfg)
    root=find_data("alchemy-aicc-round-4")
    all_train=pd.read_csv(root/"train.csv");test=pd.read_csv(root/"test.csv")
    candidates=sorted(pd.read_csv(root/"candidates.csv").result.unique().tolist())
    rng=np.random.default_rng(SEED)
    results=sorted(all_train.result.unique())
    held=set(rng.choice(results,size=max(8,len(results)//5),replace=False).tolist())
    # One row per held-out result makes the validation assignment genuinely bijective.
    validation=all_train[all_train.result.isin(held)].drop_duplicates('result').reset_index(drop=True)
    training=all_train[~all_train.result.isin(held)].reset_index(drop=True)
    train_candidates=sorted(training.result.unique())
    val_candidates=sorted(validation.result.unique())
    texts=training[['item1','item2','result']].astype(str).to_numpy().flatten().tolist()
    tokenizer=make_bpe(texts,cfg['bpe_vocab']);tokenizer.save(str(OUT/'tokenizer.json'))
    def triple(a,b,r):return '[CLS] '+str(a)+' [SEP] '+str(b)+' [SEP] '+str(r)+' [SEP]'
    def encode(rows,pool,swap=False):
        return torch.from_numpy(encode_texts(tokenizer,[triple(row.item2 if swap else row.item1,row.item1 if swap else row.item2,r)
                   for row in rows.itertuples() for r in pool],48))
    # Tokenization is deterministic and cached on CPU, not repeated in the training loop.
    tr_fwd=encode(training,train_candidates).view(len(training),len(train_candidates),48)
    tr_rev=encode(training,train_candidates,True).view_as(tr_fwd)
    val_fwd,val_rev=encode(validation,val_candidates),encode(validation,val_candidates,True)
    test_fwd,test_rev=encode(test,candidates),encode(test,candidates,True)
    model=(AlchemyComposer(tokenizer.get_vocab_size(),tied_roles=cfg.get("tied_roles",False)) if cfg.get('composer',False) else AlchemyScorer(tokenizer.get_vocab_size(),cfg['layers'])).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.03)
    scaler=torch.amp.GradScaler('cuda')
    correct=torch.tensor([train_candidates.index(v) for v in training.result])
    val_gold=np.array([val_candidates.index(v) for v in validation.result])
    def scores(ids,grad=False):
        out=[]
        for batch in ids.split(128):
            with torch.autocast('cuda',dtype=torch.float16):out.append(model(batch.to(DEVICE)).float())
        return torch.cat(out)
    def assignment_metrics(matrix):
        _,hung=linear_sum_assignment(-matrix)
        available=set(range(matrix.shape[1]));greedy=[]
        for row in matrix:
            chosen=max(available,key=lambda j:row[j]);available.remove(chosen);greedy.append(chosen)
        return dict(hungarian_accuracy=float(np.mean(hung==val_gold)),independent_accuracy=float(np.mean(matrix.argmax(1)==val_gold)),
                    greedy_accuracy=float(np.mean(np.array(greedy)==val_gold)),duplicate_argmax_count=int(len(matrix)-len(set(matrix.argmax(1)))))
    @torch.inference_mode()
    def matrix(fwd,rev,n,m):
        model.eval();a=scores(fwd).view(n,m);b=scores(rev).view(n,m)
        if cfg['probability']:combined=(a.softmax(1)+b.softmax(1))/2
        else:combined=(a+b)/2
        return combined.cpu().numpy(),float((a-b).abs().mean())
    history=[];best=-1;best_state=None;hard_scores=None;tick=time.time()
    for step in range(cfg['steps']):
        model.train()
        idx=torch.tensor(rng.choice(len(training),size=min(8,len(training)),replace=False))
        labels=correct[idx]
        pool=set(labels.tolist())
        if cfg['full']:pool=set(range(len(train_candidates)))
        else:
            if cfg['hard'] and hard_scores is not None:
                for i in idx.tolist():pool.update(hard_scores[i].argsort()[-3:].tolist())
            while len(pool)<min(16,len(train_candidates)):pool.add(int(rng.integers(len(train_candidates))))
        pool=torch.tensor(sorted(pool));targets=torch.tensor([pool.tolist().index(v) for v in labels.tolist()],device=DEVICE)
        fwd=tr_fwd[idx][:,pool].reshape(-1,48);rev=tr_rev[idx][:,pool].reshape(-1,48)
        optimizer.zero_grad(set_to_none=True)
        a=scores(fwd,True).view(len(idx),len(pool));b=scores(rev,True).view_as(a)
        combined=(a+b)/2
        loss=F.cross_entropy(combined,targets)+.05*F.mse_loss(a.softmax(1),b.softmax(1))
        if cfg['sinkhorn']:
            unique,counts=targets.unique(return_counts=True)
            if counts.max()==1:
                logp=combined
                for _ in range(5):
                    logp=logp-logp.logsumexp(1,keepdim=True)
                    logp=logp-logp.logsumexp(0,keepdim=True)
                loss=loss-.2*logp[torch.arange(len(idx),device=DEVICE),targets].mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1)
        scaler.step(optimizer);scaler.update()
        if (step+1)%40==0 or step==0:
            mat,asym=matrix(val_fwd,val_rev,len(validation),len(val_candidates));metrics=assignment_metrics(mat)
            record=dict(step=step+1,train_loss=float(loss.detach()),swap_gap=asym,**metrics)
            history.append(record);print('STEP '+json.dumps(record),flush=True)
            if metrics['hungarian_accuracy']>best:
                best=metrics['hungarian_accuracy'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if cfg['hard']:
                with torch.inference_mode():hard_scores=scores(tr_fwd.reshape(-1,48)).view(len(training),-1).cpu().numpy()
            if time.time()-tick>cfg['budget_seconds']:break
    model.load_state_dict(best_state)
    val_matrix,asym=matrix(val_fwd,val_rev,len(validation),len(val_candidates))
    test_matrix,_=matrix(test_fwd,test_rev,len(test),len(candidates))
    _,assigned=linear_sum_assignment(-test_matrix)
    sub=pd.DataFrame({'Id':test.Id,'result':[candidates[i] for i in assigned]})
    np.save(OUT/'test_scores.npy',test_matrix);np.save(OUT/'validation_scores.npy',val_matrix)
    _,vp=linear_sum_assignment(-val_matrix)
    cases=validation.copy();cases['prediction']=[val_candidates[i] for i in vp];cases['correct']=vp==val_gold
    cases.to_csv(OUT/'validation_cases.csv',index=False)
    print('VALIDATION_CASES '+cases.head(12).to_json(orient='records'),flush=True)
    torch.save(best_state,OUT/'model.pt')
    checks=dict(ids_unique=sub.Id.is_unique,schema_valid=list(sub.columns)==['Id','result'],row_count=len(sub),
                results_unique=sub.result.is_unique,allowed_results=set(sub.result)==set(candidates))
    assert checks['results_unique'] and checks['allowed_results']
    return finish_run(info,dict(metric='bijective_holdout_accuracy',value=best,direction='maximize',samples=len(validation),
            split='fixed_result_disjoint_train_holdout',**assignment_metrics(val_matrix)),sub,checks,history,
            artifacts=dict(pretrained_weights=False,tokenizer='train-only BPE',validation_cases='validation_cases.csv',
                           training_rows=len(training),heldout_result_names=val_candidates,swap_gap=asym),
            limitations=['150 rules cannot replace BERT pretraining; organizer pretrained scores are a separate condition.',
                        'Validation is small and result-disjoint; single-seed differences are uncertain.',
                        'Checkpoint selection uses holdout Hungarian accuracy, not the Kaggle test leaderboard.',
                        'This checkpoint is trained on the training fold, without a final full-data refit.'])
