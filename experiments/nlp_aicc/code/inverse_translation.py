"""Inverse Translation: memory-bounded probing and controlled discrete search."""
# Organizer reference by cowile, AI-Community-Contest/solutions (MIT).
import html, re, sys
from transformers import FSMTForConditionalGeneration

INVERSE_CONFIGS={
 1:dict(idea="Streaming top-32 probe avoids materializing the vocabulary product",topk=32,loss="margin",search="grid",restart=0),
 2:dict(idea="Top-8 proposal set trades candidate recall for speed",topk=8,loss="margin",search="grid",restart=0),
 3:dict(idea="Top-16 measures the middle recall-cost tradeoff",topk=16,loss="margin",search="grid",restart=0),
 4:dict(idea="Teacher-forced CE instead of decision margin",topk=32,loss="ce",search="grid",restart=0),
 5:dict(idea="Coarse-to-fine proposal search top-8 then top-32",topk=32,loss="margin",search="coarse",restart=0),
 6:dict(idea="Coordinate sweeps replace the full substitution grid",topk=32,loss="margin",search="coordinate",restart=0),
 7:dict(idea="Selective restarts spend compute only on non-exact rows",topk=32,loss="margin",search="coarse",restart=2)
}


def run_inverse(run_id):
    cfg={**INVERSE_CONFIGS[run_id],"probe_batch":512,"candidate_batch":64,"budget_seconds":720}
    info=start_run("inverse-translation",run_id,cfg)
    info["versions"]["transformers"]=importlib.metadata.version("transformers")
    root=find_data("inverse-translation-aicc-round-10")
    targets=json.loads((root/"targets.json").read_text())
    examples=json.loads((root/"examples.json").read_text())
    english=json.loads((root/"english_vocab.json").read_text())
    model=FSMTForConditionalGeneration.from_pretrained(root/"model").to(DEVICE).eval()
    for p in model.parameters():p.requires_grad=False
    def decode(ids):
        text="".join(english.get(str(i)," ? ") for i in ids)
        text=html.unescape(text.replace("</s>","").replace("</w>"," "))
        text=re.sub(" (?=[.,?!;:])",""," ".join(text.split()))
        return re.sub(" ?' ?","'",text)
    @torch.inference_mode()
    def generate(source,max_new):
        source_ids=F.pad(source[None],(0,1),value=2)
        output=model.generate(input_ids=source_ids,num_beams=1,max_new_tokens=max_new)[0].cpu().tolist()
        ids=[x for x in output if x!=1]
        return ids[1:] if ids and ids[0]==2 else ids
    # Verify the unmodified provided model before adapting teacher-forced positions.
    example_matches=[]
    for row in examples:
        got=generate(torch.tensor(row["alien_ids"],device=DEVICE),row["target_content_tokens"]+16)
        example_matches.append(got==row["target_ids"])
        print("EXAMPLE_CHECK "+json.dumps(dict(match=example_matches[-1],expected=row["target_ids"],generated=got,expected_text=decode(row["target_ids"]),generated_text=decode(got))),flush=True)
    if not all(example_matches):
        raise RuntimeError("Provided model generation does not match the provided examples")
    def cached_positions(module,inputs,positions):
        return positions[:,:1].expand_as(positions)
    handle=model.model.decoder.embed_positions.register_forward_hook(cached_positions)
    counters=dict(probe_batches=0,candidate_evaluations=0,model_calls=0)
    @torch.inference_mode()
    def probe():
        values=None;indices=None;tick=time.time()
        for batch in torch.arange(3,23224,device=DEVICE).split(cfg["probe_batch"]):
            source=F.pad(batch[:,None],(0,1),value=2)
            starts=torch.full((len(batch),1),2,device=DEVICE,dtype=torch.long)
            logits=model(input_ids=source,decoder_input_ids=starts,use_cache=False).logits[:,0]
            score=logits.float().log_softmax(-1).T
            local_values,local_indices=score.topk(min(cfg["topk"],len(batch)),dim=1)
            local_ids=batch[local_indices]
            if values is None:values,indices=local_values,local_ids
            else:
                joined=torch.cat([values,local_values],dim=1)
                joined_ids=torch.cat([indices,local_ids],dim=1)
                values,positions=joined.topk(cfg["topk"],dim=1)
                indices=joined_ids.gather(1,positions)
            counters["probe_batches"]+=1
            del logits,score
        return indices,time.time()-tick
    dictionary,probe_seconds=probe()
    @torch.inference_mode()
    def losses(sources,target):
        costs=[]
        for batch in sources.split(cfg["candidate_batch"]):
            labels=target.expand(len(batch),-1)
            source=F.pad(batch,(0,1),value=2)
            decoder=F.pad(labels[:,:-1],(1,0),value=2)
            logits=model(input_ids=source,decoder_input_ids=decoder,use_cache=False).logits.float()
            wanted=logits.gather(2,labels[...,None]).squeeze(-1)
            if cfg["loss"]=="margin":cost=(logits.amax(-1)-wanted).mean(1)
            else:cost=(logits.logsumexp(-1)-wanted).mean(1)
            costs.append(cost)
            counters["candidate_evaluations"]+=len(batch);counters["model_calls"]+=1
        return torch.cat(costs)
    def substitutions(source,choices):
        length,k=choices.shape
        candidates=source.repeat(length*k,1)
        rows=torch.arange(length*k,device=DEVICE)
        positions=torch.arange(length,device=DEVICE).repeat_interleave(k)
        candidates[rows,positions]=choices.flatten()
        return candidates
    def search(target,choices,source=None):
        source=choices[:,0].clone() if source is None else source.clone()
        current=float(losses(source[None],target)[0])
        for step in range(80):
            if time.time()-info["started"]>cfg["budget_seconds"]:break
            improved=False
            if cfg["search"]=="coordinate":
                for pos in range(len(source)):
                    candidates=source.repeat(choices.shape[1],1)
                    candidates[:,pos]=choices[pos]
                    costs=losses(candidates,target);best=int(costs.argmin());value=float(costs[best])
                    if value<current-1e-7:source,current=candidates[best],value;improved=True
            else:
                candidates=substitutions(source,choices)
                costs=losses(candidates,target);best=int(costs.argmin());value=float(costs[best])
                if value<current-1e-7:source,current=candidates[best],value;improved=True
            if not improved or (cfg["loss"]=="margin" and current<=1e-7):break
        return source,current
    solved={};history=[]
    for row in targets:
        target=torch.tensor(row["target_ids"],device=DEVICE)
        choices=dictionary[target[:-1]]
        if cfg["search"]=="coarse":
            source,cost=search(target,choices[:,:min(8,choices.shape[1])])
            source,cost=search(target,choices,source)
        else:source,cost=search(target,choices)
        generated=generate(source,row["target_content_tokens"]+16)
        for attempt in range(cfg["restart"]):
            if generated==row["target_ids"]:break
            random_indices=torch.randint(0,choices.shape[1],(len(source),),device=DEVICE)
            proposal=choices[torch.arange(len(source),device=DEVICE),random_indices]
            candidate,new_cost=search(target,choices,proposal)
            if new_cost<cost:
                source,cost=candidate,new_cost
                generated=generate(source,row["target_content_tokens"]+16)
        ids=source.cpu().tolist()
        assert 1<=len(ids)<=row["max_source_length"] and all(3<=x<=23223 for x in ids)
        solved[row["row_id"]]=ids
        record=dict(row_id=row["row_id"],exact=generated==row["target_ids"],
                    loss=cost,generated_text=decode(generated),source_length=len(ids))
        history.append(record);print("TARGET "+json.dumps(record),flush=True)
    handle.remove()
    sub=pd.DataFrame(dict(row_id=sorted(solved),source_ids=[json.dumps(solved[i]) for i in sorted(solved)]))
    sub.to_csv(OUT/"submission.csv",index=False)
    del model;torch.cuda.empty_cache()
    sys.path.insert(0,str(root))
    from evaluation import score_submission
    result=score_submission(root,OUT/"submission.csv")
    result={k:float(v) for k,v in result.items() if k!="rows"}
    checks=dict(ids_unique=sub.row_id.is_unique,schema_valid=list(sub.columns)==["row_id","source_ids"],
                row_count=len(sub),token_ranges_valid=True,lengths_valid=True)
    return finish_run(info,dict(metric="official_score_0_100",value=result["score"],direction="maximize",
          samples=len(targets),split="explicit_public_targets",official=result),sub,checks,history,
          artifacts=dict(probe_seconds=probe_seconds,**counters,checkpoint="provided FSMT only",
                         targets_sha256=hash_file(root/"targets.json")),
          limitations=["The target outputs are explicitly supplied by the task; this is input optimization.",
                       "Reference already reaches 100; compare cost and robustness at matched score.",
                       "Preserve provided token IDs; an unrelated BPE tokenizer would invalidate the model."])

