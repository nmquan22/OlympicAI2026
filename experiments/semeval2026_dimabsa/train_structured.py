"""Scratch span/pair extraction for DimASTE and DimASQP on English restaurant."""
import argparse
import itertools
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from prepare_data import read_jsonl, SOURCE_COMMIT
from train_scratch import fit_tokenizer, seed_everything


def build_example(row, tokenizer, categories, training=False, max_words=96, max_span=16):
    text = row["Text"]
    offsets = [match.span() for match in re.finditer(r"\S+", text)][:max_words]
    words = [text[start:end] for start, end in offsets]
    pieces = [tokenizer.encode(word.lower()).ids[:16] or [1] for word in words]
    spans = [(start, end) for start in range(len(words))
             for end in range(start, min(len(words), start + max_span))]
    index = {span: i for i, span in enumerate(spans)}
    if not training:
        return {"id": row["ID"], "text": text, "offsets": offsets, "pieces": pieces,
                "spans": spans, "relations": {}, "dropped": 0,
                "gold": row.get("Quadruplet", []), "training": False}
    starts = {start: i for i, (start, end) in enumerate(offsets)}
    ends = {end: i for i, (start, end) in enumerate(offsets)}
    pairs = defaultdict(list)
    dropped = 0
    for quad in row.get("Quadruplet", []):
        if quad["Aspect"] == "NULL" or quad["Opinion"] == "NULL":
            continue
        matches = []
        for field in ("Aspect", "Opinion"):
            found = [(starts[m.start()], ends[m.end()])
                     for m in re.finditer(re.escape(quad[field]), text, flags=re.I)
                     if m.start() in starts and m.end() in ends]
            matches.append([span for span in found if span in index])
        if not matches[0] or not matches[1]:
            dropped += 1
            continue
        aspect, opinion = min(itertools.product(*matches), key=lambda pair: abs(pair[0][0] - pair[1][0]))
        pairs[index[aspect], index[opinion]].append(quad)
    relations = {}
    for pair, quads in pairs.items():
        va = np.mean([[float(value) for value in q["VA"].split("#")] for q in quads], axis=0).tolist()
        relations[pair] = (categories[quads[0]["Category"]], va)
    return {"id": row["ID"], "text": text, "offsets": offsets, "pieces": pieces,
            "spans": spans, "relations": relations, "dropped": dropped,
            "gold": row.get("Quadruplet", []), "training": training}


def collate(examples):
    words = max(len(example["pieces"]) for example in examples)
    pieces = max(len(ids) for example in examples for ids in example["pieces"])
    batch = torch.zeros(len(examples), words, pieces, dtype=torch.long)
    lengths = torch.tensor([len(e["pieces"]) for e in examples])
    for i, example in enumerate(examples):
        for j, ids in enumerate(example["pieces"]):
            batch[i, j, :len(ids)] = torch.tensor(ids)
    return batch, lengths, examples


def balanced_bce(logits, targets):
    positive = targets.bool()
    terms = []
    if positive.any():
        terms.append(F.softplus(-logits[positive]).mean())
    if (~positive).any():
        terms.append(F.softplus(logits[~positive]).mean())
    return torch.stack(terms).mean()


class SpanPairModel(nn.Module):
    def __init__(self, vocabulary, category_count, width=192, max_span=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.encoder = nn.LSTM(width, width // 2, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=0.25)
        self.dropout = nn.Dropout(0.25)
        self.length = nn.Embedding(max_span + 1, 16)
        self.distance = nn.Embedding(129, 16)
        self.span_projection = nn.Sequential(nn.Linear(width * 3 + 16, width), nn.GELU(), nn.Dropout(0.25))
        self.span_head = nn.Linear(width, 2)
        self.pair_projection = nn.Sequential(nn.Linear(width * 4 + 16, width), nn.GELU(), nn.Dropout(0.25))
        self.link_head = nn.Linear(width, 1)
        self.va_head = nn.Linear(width, 2)
        self.category_head = nn.Linear(width, category_count)

    def encode(self, ids, lengths):
        mask = ids.ne(0).unsqueeze(-1)
        words = (self.embedding(ids) * mask).sum(2) / mask.sum(2).clamp_min(1)
        packed = pack_padded_sequence(self.dropout(words), lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=ids.shape[1])
        return hidden

    def span_features(self, hidden, spans):
        spans = torch.as_tensor(spans, device=hidden.device)
        start, end = spans[:, 0], spans[:, 1]
        prefix = torch.cat([hidden.new_zeros(1, hidden.shape[-1]), hidden.cumsum(0)], dim=0)
        mean = (prefix[end + 1] - prefix[start]) / (end - start + 1).unsqueeze(-1)
        features = torch.cat([hidden[start], hidden[end], mean, self.length(end - start + 1)], dim=-1)
        projected = self.span_projection(features)
        return projected, self.span_head(projected)

    def pair_predictions(self, span_hidden, spans, aspect_ids, opinion_ids):
        device = span_hidden.device
        span_positions = torch.as_tensor(spans, device=device)
        pairs = list(itertools.product(aspect_ids, opinion_ids))
        pair_ids = torch.as_tensor(pairs, device=device)
        aspect, opinion = span_hidden[pair_ids[:, 0]], span_hidden[pair_ids[:, 1]]
        distances = (span_positions[pair_ids[:, 1], 0] - span_positions[pair_ids[:, 0], 0]).clamp(-64, 64) + 64
        features = torch.cat([aspect, opinion, aspect * opinion, torch.abs(aspect - opinion),
                              self.distance(distances)], dim=-1)
        hidden = self.pair_projection(features)
        return pairs, self.link_head(hidden).squeeze(-1), 1 + 8 * self.va_head(hidden).sigmoid(), self.category_head(hidden)

    def example_loss(self, hidden, example):
        span_hidden, logits = self.span_features(hidden, example["spans"])
        device = hidden.device
        gold = example["relations"]
        aspects = sorted({pair[0] for pair in gold})
        opinions = sorted({pair[1] for pair in gold})
        positives = sorted(set(aspects + opinions))
        negatives = list(set(range(len(example["spans"]))) - set(positives))
        selected = positives + random.sample(negatives, min(80, len(negatives)))
        targets = torch.tensor([[i in aspects, i in opinions] for i in selected], device=device, dtype=torch.float32)
        span_loss = sum(balanced_bce(logits[selected, k], targets[:, k]) for k in range(2))
        # Gold spans guarantee positives; detached predicted spans supply hard negatives.
        k = min(8, len(example["spans"]))
        candidate_a = sorted(set(aspects + logits[:, 0].detach().topk(k).indices.tolist()))
        candidate_o = sorted(set(opinions + logits[:, 1].detach().topk(k).indices.tolist()))
        pairs, links, va, categories = self.pair_predictions(span_hidden, example["spans"], candidate_a, candidate_o)
        links_gold = torch.tensor([pair in gold for pair in pairs], device=device, dtype=torch.float32)
        link_loss = balanced_bce(links, links_gold)
        positive_ids = [i for i, pair in enumerate(pairs) if pair in gold]
        category_gold = torch.tensor([gold[pairs[i]][0] for i in positive_ids], device=device)
        va_gold = torch.tensor([gold[pairs[i]][1] for i in positive_ids], device=device)
        va_loss = F.mse_loss(va[positive_ids] / 4, va_gold / 4)
        category_loss = F.cross_entropy(categories[positive_ids], category_gold)
        return span_loss + link_loss + va_loss + 0.5 * category_loss


@torch.inference_mode()
def cache_predictions(model, loader, category_names, device):
    model.eval()
    cache = []
    for ids, lengths, examples in loader:
        hidden = model.encode(ids.to(device), lengths)
        for i, example in enumerate(examples):
            span_hidden, logits = model.span_features(hidden[i, :lengths[i]], example["spans"])
            probabilities = logits.sigmoid()
            k = min(8, len(example["spans"]))
            aspects = probabilities[:, 0].topk(k).indices.tolist()
            opinions = probabilities[:, 1].topk(k).indices.tolist()
            pairs, links, va, categories = model.pair_predictions(span_hidden, example["spans"], aspects, opinions)
            links, va = links.sigmoid().cpu().tolist(), va.cpu().tolist()
            categories = categories.argmax(-1).cpu().tolist()
            probabilities = probabilities.cpu().numpy()
            candidates = []
            for j, (a, o) in enumerate(pairs):
                a_start, a_end = example["spans"][a]
                o_start, o_end = example["spans"][o]
                offsets, text = example["offsets"], example["text"]
                candidates.append({"Aspect": text[offsets[a_start][0]:offsets[a_end][1]],
                                   "Opinion": text[offsets[o_start][0]:offsets[o_end][1]],
                                   "Category": category_names[categories[j]],
                                   "VA": f"{va[j][0]:.2f}#{va[j][1]:.2f}",
                                   "span_probability": float(min(probabilities[a, 0], probabilities[o, 1])),
                                   "link_probability": links[j]})
            cache.append(candidates)
    return cache


def decode(candidates, span_threshold, link_threshold, task):
    unique = {}
    for candidate in candidates:
        if candidate["span_probability"] < span_threshold or candidate["link_probability"] < link_threshold:
            continue
        fields = ("Aspect", "Opinion", "Category") if task == 3 else ("Aspect", "Opinion")
        key = tuple(candidate[field].lower() for field in fields)
        score = candidate["span_probability"] * candidate["link_probability"]
        if key not in unique or score > unique[key][0]:
            unique[key] = (score, {field: candidate[field] for field in (*fields, "VA")})
    return [value[1] for value in unique.values()]


def continuous_f1(gold_rows, predictions, task):
    """Official formula for valid, duplicate-free predictions; recheck final files with BTC."""
    fields = ("Aspect", "Opinion", "Category") if task == 3 else ("Aspect", "Opinion")
    total_gold, total_pred, continuous_tp, exact_tp = 0, 0, 0.0, 0
    for row, predicted in zip(gold_rows, predictions):
        gold = row.get("Quadruplet", row.get("Triplet", []))
        total_gold += len(gold)
        total_pred += len(predicted)
        indexed = {tuple(q[field].lower() for field in fields): q for q in predicted}
        for item in gold:
            key = tuple(item[field].lower() for field in fields)
            if key in indexed:
                y = np.array([float(v) for v in item["VA"].split("#")])
                p = np.array([float(v) for v in indexed[key]["VA"].split("#")])
                continuous_tp += max(0.0, 1 - float(np.linalg.norm(p - y)) / 128 ** 0.5)
                exact_tp += 1
    denominator = max(1, total_gold + total_pred)
    return {"cF1": 2 * continuous_tp / denominator, "tuple_F1_without_VA": 2 * exact_tp / denominator,
            "predicted_tuples": total_pred, "gold_tuples": total_gold, "matched_tuples": exact_tp}


def evaluate_cache(cache, gold_rows):
    best = None
    for span_threshold, link_threshold in itertools.product((0.3, 0.5, 0.7), repeat=2):
        decoded = {task: [decode(candidates, span_threshold, link_threshold, task) for candidates in cache]
                   for task in (2, 3)}
        scores = {task: continuous_f1(gold_rows, decoded[task], task) for task in (2, 3)}
        selection = (scores[2]["cF1"] + scores[3]["cF1"]) / 2
        result = {"selection_score": selection, "span_threshold": span_threshold, "link_threshold": link_threshold,
                  "subtask_2": scores[2], "subtask_3": scores[3]}
        if best is None or selection > best[0]["selection_score"]:
            best = result, decoded
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    seed_everything(42)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train_rows = read_jsonl(args.data / "eng_restaurant_train_alltasks.jsonl")
    dev_rows = read_jsonl(args.data / "eng_restaurant_dev_task3.jsonl")
    dev2_rows = read_jsonl(args.data / "eng_restaurant_dev_task2.jsonl")
    if [row["Text"] for row in dev_rows] != [row["Text"] for row in dev2_rows]:
        raise ValueError("Dev task 2 and task 3 sentence order differs")
    dev_texts = {" ".join(r["Text"].lower().split()) for r in dev_rows}
    train_rows = [row for row in train_rows if " ".join(row["Text"].lower().split()) not in dev_texts]
    category_names = sorted({q["Category"] for row in train_rows for q in row["Quadruplet"]})
    categories = {name: i for i, name in enumerate(category_names)}
    tokenizer = fit_tokenizer([{"text": row["Text"]} for row in train_rows], 3000)
    tokenizer.save(str(args.out / "tokenizer.json"))
    examples = [build_example(row, tokenizer, categories, training=True) for row in train_rows]
    dropped = sum(e["dropped"] for e in examples)
    train = [e for e in examples if e["relations"]]
    dev = [build_example(row, tokenizer, categories) for row in dev_rows]
    train_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=16, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpanPairModel(tokenizer.get_vocab_size(), len(categories)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)
    best, stale, best_epoch, best_state, best_config, best_predictions = -1, 0, 0, None, None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, lengths, batch_examples in train_loader:
            optimizer.zero_grad(set_to_none=True)
            hidden = model.encode(ids.to(device), lengths)
            loss = torch.stack([model.example_loss(hidden[i, :lengths[i]], example)
                                for i, example in enumerate(batch_examples)]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite structured training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        cached = cache_predictions(model, dev_loader, category_names, device)
        result, predictions = evaluate_cache(cached, dev_rows)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.monotonic() - started, **result}
        print(json.dumps(entry), flush=True)
        with (args.out / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        if result["selection_score"] > best:
            best, best_epoch, stale = result["selection_score"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_config, best_predictions = result, predictions
            torch.save({"state_dict": best_state, "categories": category_names}, args.out / "best.pt")
            (args.out / "dev_candidates.json").write_text(json.dumps(cached, ensure_ascii=False))
        else:
            stale += 1
        if stale >= 5 or time.monotonic() - started >= args.seconds:
            break
    for task, rows in ((2, dev2_rows), (3, dev_rows)):
        key = {2: "Triplet", 3: "Quadruplet"}[task]
        lines = [json.dumps({"ID": row["ID"], key: tuples}, ensure_ascii=False)
                 for row, tuples in zip(rows, best_predictions[task])]
        (args.out / f"dev_task{task}_predictions.jsonl").write_text("\n".join(lines) + "\n")
    report = {"run": "S01", "status": "completed", "architecture": "random BPE word pooling + BiLSTM + span/pair/category/VA heads",
              "pretrained_weights": False, "source_commit": SOURCE_COMMIT, "train_sentences": len(train),
              "train_relations": sum(len(e["relations"]) for e in train), "dropped_training_quads": dropped,
              "word_limit": 96, "span_word_limit": 16, "best_epoch": best_epoch,
              "seed": 42, "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic() - started,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "torch_version": torch.__version__,
              "test_evaluated": False, "categories": category_names, **best_config}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
