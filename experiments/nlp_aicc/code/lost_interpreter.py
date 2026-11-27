"""Lost Interpreter: random-init Transformer, BPE, structured and numeric ablations."""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re


LOST_CONFIGS={
 1:dict(idea="BPE Transformer with mean/max pooling and direct MAE loss",structure=False,numeric=False,attention=False,aux=False,target="standard",layers=2),
 2:dict(idea="Expose instruction boundaries using dedicated separators",structure=True,numeric=False,attention=False,aux=False,target="standard",layers=2),
 3:dict(idea="Numeric summary branch supplements subword representations",structure=True,numeric=True,attention=False,aux=False,target="standard",layers=2),
 4:dict(idea="Learned query attention pooling replaces fixed pooling",structure=True,numeric=False,attention=True,aux=False,target="standard",layers=2),
 5:dict(idea="Auxiliary instruction-count prediction regularizes program structure",structure=True,numeric=False,attention=False,aux=True,target="standard",layers=2),
 6:dict(idea="Signed-log target tests outlier compression under MAE",structure=True,numeric=False,attention=False,aux=False,target="signed_log",layers=2),
 7:dict(idea="Combine structured tokens, numeric branch and query pooling",structure=True,numeric=True,attention=True,aux=True,target="standard",layers=2)
}


LOST_CONFIGS[8]={**LOST_CONFIGS[1],"idea":"Scratch BPE BiGRU with packed sequences, masked pooling and SmoothL1","backbone":"bigru","loss":"smooth_l1"}

def program_text(value,structured):
    text=str(value).replace(";"," ; ")
    if structured:
        text=text.replace(";"," [SEP] ")
        text=re.sub(r"(-?\d+)",r" [NUM] \1 ",text)
    return "[CLS] "+text


def numeric_features(program):
    numbers=np.array([float(n) for n in re.findall(r"(?<![A-Za-z])[-+]?\d+",str(program))],dtype=float)
    if not len(numbers):numbers=np.array([0.])
    return np.array([len(str(program).split(";")),len(numbers),numbers.sum(),numbers.mean(),
                     numbers.min(),numbers.max(),numbers.std(),np.abs(numbers).sum()],dtype=np.float32)


class ProgramTransformer(nn.Module):
    def __init__(self,vocab,config,dim=96):
        super().__init__();self.config=config
        self.encoder=PositionEncoder(vocab,dim,config["layers"],max_length=256)
        if config["attention"]:
            self.query=nn.Parameter(torch.randn(dim)/math.sqrt(dim))
        self.numerical=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.LayerNorm(32))
        width=dim*2+(32 if config["numeric"] else 0)
        self.head=nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,1))
        self.auxiliary=nn.Linear(dim*2,1)

    def forward(self,ids,numbers):
        h,mask=self.encoder(ids)
        representation=pools(h,mask)
        if self.config["attention"]:
            scores=(h*self.query).sum(-1)/math.sqrt(h.shape[-1])
            weights=scores.masked_fill(mask,-1e4).softmax(-1)
            pooled=(h*weights.unsqueeze(-1)).sum(1)
            representation=torch.cat([pooled,representation[:,h.shape[-1]:]],-1)
        final=torch.cat([representation,self.numerical(numbers)],-1) if self.config["numeric"] else representation
        return self.head(final).squeeze(-1),self.auxiliary(representation).squeeze(-1)


class ProgramBiGRU(nn.Module):
    def __init__(self,vocab,dim=128):
        super().__init__()
        self.embedding=nn.Embedding(vocab,dim,padding_idx=0)
        self.gru=nn.GRU(dim,dim,2,batch_first=True,bidirectional=True,dropout=.1)
        self.head=nn.Sequential(nn.Linear(dim*4,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
        self.auxiliary=nn.Linear(dim*4,1)
    def forward(self,ids,numbers):
        lengths=ids.ne(0).sum(1).clamp(min=1).cpu()
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(ids),lengths,batch_first=True,enforce_sorted=False)
        encoded,_=self.gru(packed)
        h,_=nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=ids.shape[1])
        representation=pools(h,ids.eq(0))
        return self.head(representation).squeeze(-1),self.auxiliary(representation).squeeze(-1)


def run_lost(run_id):
    config={**LOST_CONFIGS[run_id],"epochs":45,"budget_seconds":300,"batch_size":128,"bpe_vocab":2048,"max_length":192}
    info=start_run("lost-interpreter",run_id,config)
    root=find_data("lost-interpreter-aicc-round-5")
    train=pd.read_csv(root/"train.csv");test=pd.read_csv(root/"test.csv")
    assert {"program","output"}.issubset(train.columns) and {"ID","program"}.issubset(test.columns)
    tr_idx,va_idx=train_test_split(np.arange(len(train)),test_size=0.15,random_state=SEED)
    if run_id>=8:print("TRAIN_PROGRAM_EXAMPLES "+train.iloc[tr_idx[:8]][["program","output"]].to_json(orient="records"),flush=True)
    texts=[program_text(p,config["structure"]) for p in train.program]
    test_texts=[program_text(p,config["structure"]) for p in test.program]
    tokenizer=make_bpe((texts[i] for i in tr_idx),config["bpe_vocab"])
    tokenizer.add_special_tokens(["[NUM]"])
    tokenizer.save(str(OUT/"tokenizer.json"))
    ids=encode_texts(tokenizer,texts,config["max_length"])
    test_ids=encode_texts(tokenizer,test_texts,config["max_length"])
    num=np.stack([numeric_features(p) for p in train.program])
    test_num=np.stack([numeric_features(p) for p in test.program])
    num=np.sign(num)*np.log1p(np.abs(num));test_num=np.sign(test_num)*np.log1p(np.abs(test_num))
    center,scale=num[tr_idx].mean(0),np.maximum(num[tr_idx].std(0),1e-6)
    num=(num-center)/scale;test_num=(test_num-center)/scale
    truth=train.output.to_numpy(dtype=np.float32)
    transformed=np.sign(truth)*np.log1p(np.abs(truth)) if config["target"]=="signed_log" else truth
    ym,ys=float(transformed[tr_idx].mean()),max(float(transformed[tr_idx].std()),1e-6)
    target=(transformed-ym)/ys
    aux=np.array([len(str(p).split(";")) for p in train.program],dtype=np.float32)
    aux=np.log1p(aux)
    am,astd=float(aux[tr_idx].mean()),max(float(aux[tr_idx].std()),1e-6)
    aux=(aux-am)/astd
    tr=TensorDataset(torch.from_numpy(ids[tr_idx]),torch.from_numpy(num[tr_idx]),
                     torch.from_numpy(target[tr_idx]),torch.from_numpy(aux[tr_idx]))
    model=(ProgramBiGRU(tokenizer.get_vocab_size()) if config.get("backbone")=="bigru" else ProgramTransformer(tokenizer.get_vocab_size(),config)).to(DEVICE)
    optim=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=0.01)
    scaler=torch.amp.GradScaler("cuda")
    loader=DataLoader(tr,batch_size=config["batch_size"],shuffle=True,num_workers=0)
    def predict_ids(x,n):
        model.eval();values=[]
        with torch.inference_mode():
            for a,b in DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(n)),batch_size=192):
                with torch.autocast("cuda",dtype=torch.float16):
                    p,_=model(a.to(DEVICE),b.to(DEVICE))
                values.append(p.float().cpu().numpy())
        raw=np.concatenate(values)*ys+ym
        if config["target"]=="signed_log":raw=np.sign(raw)*np.expm1(np.clip(np.abs(raw),0,20))
        return raw
    best=float("inf");best_state=None;history=[];bad=0;tick=time.time()
    for epoch in range(config["epochs"]):
        model.train();losses=[]
        for x,n,y,a in loader:
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                pred,aux_pred=model(x.to(DEVICE),n.to(DEVICE))
                loss=F.smooth_l1_loss(pred,y.to(DEVICE),beta=.2) if config.get("loss")=="smooth_l1" else F.l1_loss(pred,y.to(DEVICE))
                if config["aux"]:loss=loss+0.05*F.mse_loss(aux_pred,a.to(DEVICE))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optim);scaler.update()
            losses.append(float(loss.detach()))
        pred=predict_ids(ids[va_idx],num[va_idx])
        mae=float(mean_absolute_error(truth[va_idx],pred))
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_mae=mae)
        history.append(row);print("EPOCH "+json.dumps(row),flush=True)
        if mae<best:
            best=mae;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=9 or time.time()-tick>config["budget_seconds"]:break
    model.load_state_dict(best_state)
    val=predict_ids(ids[va_idx],num[va_idx]);pred=predict_ids(test_ids,test_num)
    assert np.isfinite(pred).all()
    validation_cases=pd.DataFrame(dict(row_index=va_idx,program=train.program.iloc[va_idx].tolist(),truth=truth[va_idx],prediction=val))
    validation_cases["absolute_error"]=np.abs(val-truth[va_idx])
    validation_cases["instructions"]=validation_cases.program.map(lambda p:len(str(p).split(";")))
    validation_cases.to_csv(OUT/"validation_predictions.csv",index=False)
    print("VALIDATION_WORST "+validation_cases.nlargest(8,"absolute_error").to_json(orient="records"),flush=True)
    by_length=validation_cases.groupby(pd.cut(validation_cases.instructions,[0,4,8,16,10000]),observed=True).absolute_error.agg(["mean","count"])
    print("ERROR_BY_LENGTH "+by_length.to_json(),flush=True)
    pd.DataFrame(dict(row_index=tr_idx)).to_csv(OUT/"train_indices.csv",index=False)
    torch.save(dict(state=best_state,y_mean=ym,y_std=ys,numeric_mean=center,numeric_std=scale),OUT/"model.pt")
    sub=pd.DataFrame(dict(ID=test.ID,output=np.rint(pred).astype(np.int64)))
    checks=dict(ids_unique=sub.ID.is_unique,schema_valid=list(sub.columns)==["ID","output"],
                row_count=len(sub),ids_match=bool(sub.ID.equals(test.ID)),finite=bool(np.isfinite(pred).all()))
    return finish_run(info,dict(metric="MAE",value=best,direction="minimize",samples=len(va_idx),
          split="fixed_train_holdout_15pct"),sub,checks,history,
          artifacts=dict(tokenizer="tokenizer.json",pretrained_weights=False,
                         train_sha256=hash_file(root/"train.csv"),test_sha256=hash_file(root/"test.csv")),
          limitations=["Single fixed holdout. Results depend on optimization budget.",
                       "The official source uses a different split; its MAE is not a paired comparison."])

