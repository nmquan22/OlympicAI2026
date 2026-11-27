"""Tally: scratch BPE, word-aligned NER and entity-count supervision.

No external data, model weights, test counts or pretrained tokenizer. All metric
decisions use a fixed 40/10 gold split; ten sentences are only a noisy pilot.
"""
from collections import Counter
from sklearn.model_selection import train_test_split
import string

TALLY_LABELS = ['O', 'PERSON', 'ORGANIZATION', 'LOCATION', 'CREATIVE_WORK', 'EVENT']
TALLY_CONFIGS = {
    1: dict(idea='Scratch byte BPE word pooling + packed BiGRU, gold CE and expected entity-run counts',
            objective='expected_run_counts', epochs=35, budget_seconds=480, batch_size=64,
            gold_batch_size=16, warmup_steps=160, bpe_vocab=2048, dim=128,
            weak_weight=.5, lr=.001, patience=8)
}


def tally_spans(tags):
    """IO labels: contiguous same non-O type is exactly one scored entity."""
    result = set(); start = 0
    for t in range(len(tags) + 1):
        if t == len(tags) or tags[t] != tags[start]:
            if tags[start] not in (0, 'O'):
                result.add((start, t, tags[start]))
            start = t
    return result


def tally_counts(tags):
    count = np.zeros(5, dtype=np.int64)
    for _, _, c in tally_spans(tags):
        count[(TALLY_LABELS.index(c) if isinstance(c, str) else c) - 1] += 1
    return count


def tally_metrics(gold, predictions, texts, train_words):
    tp = fp = fn = 0; support = Counter(); correct = Counter(); predicted = Counter()
    failures = Counter(); cases = []; unseen_tp = unseen_fn = 0
    for truth, pred, text in zip(gold, predictions, texts):
        g, p = tally_spans(truth), tally_spans(pred); words = text.split()
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        for a,b,c in g:
            support[TALLY_LABELS[c]] += 1
            if any(w not in train_words for w in words[a:b]):
                unseen_tp += int((a,b,c) in p); unseen_fn += int((a,b,c) not in p)
        for a,b,c in p: predicted[TALLY_LABELS[c]] += 1
        for a,b,c in g & p: correct[TALLY_LABELS[c]] += 1
        # Categories describe missed gold spans and need not sum to false positives.
        for a,b,c in g-p:
            if any(a == d and b == e for d,e,_ in p): failures['correct_boundary_wrong_type'] += 1
            elif any(c == f and a < e and d < b for d,e,f in p): failures['overlap_right_type_wrong_boundary'] += 1
            else: failures['no_right_type_overlap'] += 1
        if not p: failures['all_O_sentences'] += 1
        for a,b,c in p:
            if not tally_start_allowed(words[a]): failures['illegal_entity_start'] += 1
            if not tally_end_allowed(words[b-1]): failures['illegal_entity_end'] += 1
        if p != g:
            cases.append(dict(text=text, truth=' '.join(TALLY_LABELS[t] for t in truth),
                              prediction=' '.join(TALLY_LABELS[t] for t in pred),
                              missed=[list(v) for v in sorted(g-p)],
                              spurious=[list(v) for v in sorted(p-g)]))
    f1 = 2*tp/max(1, 2*tp+fp+fn)
    return dict(f1=f1, true_positive=tp, false_positive=fp, false_negative=fn,
                per_type={c:dict(gold=support[c], predicted=predicted[c], correct=correct[c],
                    f1=2*correct[c]/max(1,support[c]+predicted[c])) for c in TALLY_LABELS[1:]},
                failure_modes=dict(failures), unseen_word_entity_recall=unseen_tp/max(1,unseen_tp+unseen_fn),
                unseen_word_gold_entities=unseen_tp+unseen_fn, cases=cases)


def tally_start_allowed(word):
    return word not in {'the', 'a', 'an'} and not all(c in string.punctuation for c in word)


def tally_end_allowed(word):
    return word != "'s" and not all(c in string.punctuation for c in word)


def expected_tally_counts(logits, mask):
    # Exact under an independent per-word label distribution. This is NOT a CRF
    # pair marginal, and a count is a transition into a type, not token mass.
    p = logits.float().softmax(-1)[..., 1:]
    previous = torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], dim=1)
    return (p * (1-previous) * mask.unsqueeze(-1)).sum(1)


class TallyModel(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.gru = nn.GRU(dim+2, dim, 2, batch_first=True, bidirectional=True, dropout=.15)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(dim*2, 6))

    def forward(self, pieces, features, mask):
        keep = pieces.ne(0).unsqueeze(-1)
        h = (self.embedding(pieces)*keep).sum(2)/keep.sum(2).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(torch.cat([h, features], -1),
                    mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=pieces.shape[1])
        return self.head(h)


def run_tally(run_id):
    config = deepcopy(TALLY_CONFIGS[run_id]); info = start_run('tally', run_id, config)
    root = find_data('tally-aicc-round-11')
    weak = pd.read_csv(root/'train_counts.csv'); gold = pd.read_csv(root/'train_tags.csv')
    test = pd.read_csv(root/'test.csv'); sample = pd.read_csv(root/'sample_submission.csv')
    assert len(gold) == 50 and len(weak) == 8000 and len(test) == 4000
    assert {'id','text',*TALLY_LABELS[1:]}.issubset(weak.columns)
    assert gold.id.is_unique and weak.id.is_unique and test.id.is_unique
    truth = [[TALLY_LABELS.index(t) for t in value.split()] for value in gold.tags]
    assert all(len(t)==len(s.split()) for t,s in zip(truth,gold.text))
    tr, va = train_test_split(np.arange(50), test_size=10, random_state=SEED)
    wtr, wva = train_test_split(np.arange(len(weak)), test_size=800, random_state=SEED)
    heldout_texts = set(gold.text.iloc[va])
    wtr = np.array([i for i in wtr if weak.text.iloc[i] not in heldout_texts])
    train_texts = weak.text.iloc[wtr].tolist()+gold.text.iloc[tr].tolist()
    assert not heldout_texts.intersection(train_texts)
    freq = Counter(w for text in train_texts for w in text.split())
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(train_texts, trainers.BpeTrainer(vocab_size=config['bpe_vocab'],
        min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['[PAD]','[UNK]','[MASK]']))
    tok.save(str(OUT/'tokenizer.json'))
    cache = {}
    def encode(text):
        words = text.split()
        for word in words:
            if word not in cache: cache[word] = tok.encode(word).ids
        return dict(text=text, words=words, pieces=[cache[w] for w in words],
                    features=[[math.log1p(freq[w])/8, math.log1p(len(w))/3] for w in words])
    gx = [encode(s) for s in gold.text]; wx = [encode(s) for s in weak.text]
    tx = [encode(s) for s in test.text]
    def batch(examples):
        b = len(examples); length = max(len(x['words']) for x in examples)
        parts = max(len(p) for x in examples for p in x['pieces'])
        ids = torch.zeros(b,length,parts,dtype=torch.long)
        ft = torch.zeros(b,length,2); mask = torch.zeros(b,length,dtype=torch.bool)
        for j,x in enumerate(examples):
            n = len(x['words']); mask[j,:n] = True; ft[j,:n] = torch.tensor(x['features'])
            for t, p in enumerate(x['pieces']): ids[j,t,:len(p)] = torch.tensor(p)
        return ids.to(DEVICE), ft.to(DEVICE), mask.to(DEVICE)
    def gold_loss(indices):
        examples = [gx[i] for i in indices]; ids,ft,mask = batch(examples)
        labels = torch.full(mask.shape,-100,dtype=torch.long,device=DEVICE)
        for j,i in enumerate(indices): labels[j,:len(truth[i])] = torch.tensor(truth[i],device=DEVICE)
        logits = model(ids,ft,mask)
        return F.cross_entropy(logits.float().reshape(-1,6), labels.reshape(-1))
    def predict(examples):
        result = []; model.eval()
        with torch.inference_mode():
            for start in range(0,len(examples),128):
                rows = examples[start:start+128]; ids,ft,mask = batch(rows)
                with torch.autocast('cuda', dtype=torch.float16): logits = model(ids,ft,mask)
                pred = logits.argmax(-1).cpu().tolist()
                result += [p[:len(r['words'])] for p,r in zip(pred,rows)]
        return result
    counts = weak[TALLY_LABELS[1:]].to_numpy(dtype=np.float32)
    model = TallyModel(tok.get_vocab_size(), config['dim']).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda'); history = []; tick = time.time()
    def update(loss):
        assert bool(torch.isfinite(loss)), 'Non-finite training loss'
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),1); scaler.step(opt); scaler.update()
    for _ in range(config['warmup_steps']):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss = gold_loss(np.random.choice(tr,16))
        update(loss)
    best = -1.; state = None; bad = 0; best_epoch = 0; steps = 0
    for epoch in range(config['epochs']):
        losses = []; order = np.random.permutation(wtr)
        for start in range(0,len(order),config['batch_size']):
            model.train(); opt.zero_grad(set_to_none=True)
            ix = order[start:start+config['batch_size']]
            ids,ft,mask = batch([wx[i] for i in ix])
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(ids,ft,mask)
                expected = expected_tally_counts(logits,mask)
                target = torch.tensor(counts[ix],device=DEVICE)
                # Scale high-count rows without turning zero labels into missing labels.
                count_loss = (((expected-target)**2)/(target+1)).mean()
                loss = gold_loss(np.random.choice(tr,config['gold_batch_size'])) + config['weak_weight']*count_loss
            update(loss); losses.append(float(loss.detach())); steps += 1
            if time.time()-tick > config['budget_seconds']: break
        pred = predict([gx[i] for i in va])
        measured = tally_metrics([truth[i] for i in va],pred,gold.text.iloc[va].tolist(),freq)
        row = dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_entity_f1=measured['f1'],
                   elapsed_training_seconds=time.time()-tick)
        history.append(row); print('EPOCH '+json.dumps(row),flush=True)
        if measured['f1'] > best:
            best = measured['f1']; best_epoch = epoch+1; bad = 0
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad += 1
        if bad >= config['patience'] or time.time()-tick > config['budget_seconds']: break
    assert state is not None
    model.load_state_dict(state); torch.save(dict(model=state,config=config),OUT/'model.pt')
    valpred = predict([gx[i] for i in va])
    measured = tally_metrics([truth[i] for i in va],valpred,gold.text.iloc[va].tolist(),freq)
    weakpred = predict([wx[i] for i in wva])
    weakcounts = np.stack([tally_counts(p) for p in weakpred]); delta = weakcounts-counts[wva]
    diagnostic = {k:v for k,v in measured.items() if k != 'cases'}
    diagnostic.update(weak_holdout_count_mae=float(np.abs(delta).mean()),
        weak_holdout_exact_count_vector=float((delta==0).all(1).mean()),
        weak_holdout_samples=len(wva), best_epoch=best_epoch,optimizer_steps=steps+config['warmup_steps'],
        gold_train_ids=gold.id.iloc[tr].tolist(),gold_validation_ids=gold.id.iloc[va].tolist(),
        per_type_gold_train_support=dict(Counter(TALLY_LABELS[c] for i in tr for _,_,c in tally_spans(truth[i]))),
        weak_train_samples=len(wtr), max_words=max(len(x['words']) for x in gx+wx+tx),
        bpe_vocab=tok.get_vocab_size(),test_unk_pieces=sum(p==1 for x in tx for ids in x['pieces'] for p in ids),
        test_total_pieces=sum(len(ids) for x in tx for ids in x['pieces']),
        validation_cases=measured['cases'])
    (OUT/'failure_analysis.json').write_text(json.dumps(diagnostic,indent=2))
    print('TALLY_FAILURE_ANALYSIS '+json.dumps(diagnostic),flush=True)
    testpred = predict(tx)
    submission = pd.DataFrame(dict(id=test.id,tags=[' '.join(TALLY_LABELS[c] for c in p) for p in testpred]))
    assert list(submission.id)==list(test.id) and set(submission.id)==set(sample.id)
    assert all(len(tags.split())==len(text.split()) for tags,text in zip(submission.tags,test.text))
    assert set(' '.join(submission.tags).split()).issubset(TALLY_LABELS)
    info['diagnostics'] = diagnostic
    info['data_provenance'] = {p.name:hash_file(p) for p in [root/'train_counts.csv',root/'train_tags.csv',root/'test.csv']}
    return finish_run(info, dict(metric='entity_span_micro_F1',value=measured['f1'],direction='maximize',
        samples=len(va),split='fixed_gold_40_train_10_validation_seed42'), submission,
        dict(ids_unique=bool(submission.id.is_unique),schema_valid=True,row_count=len(submission),
             word_tag_alignment=True,allowed_labels_only=True),history,
        artifacts=dict(failure_analysis='failure_analysis.json',model='model.pt',tokenizer='tokenizer.json'),
        limitations=['Only ten gold validation sentences; no significance claim or SOTA comparison.',
                     'Selected checkpoint uses 40 gold training sentences; no full-50 refit.',
                     'Expected run count is exact only for independent word labels, not CRF marginals.',
                     'No test count labels used; all weights and tokenizer trained from scratch.'])
