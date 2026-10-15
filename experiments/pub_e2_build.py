# Decision dataset builder for PUBLIC corpora (cases.jsonl -> e2 splits).
# Rules are a verbatim adaptation of e2_build.py (split_of / fold_events /
# features / train-graph + edge vocab), so public and SOC decision sets are
# definition-identical. Lineage self-test: local fold_events must agree with
# fold_rules.fold kind-for-kind on random sequences.
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_rules import fold  # noqa: E402

# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container
_host = Path("/lw2/pubdata")
PUB = _host if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
STOP = "<stop>"


def split_of(trace):
    h = int(hashlib.sha256(trace.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 80 else "test")


def fold_events(labels):
    """Yield (kind, kept_prefix_labels_after, label, prev) per step per §3 rules."""
    kept = []
    evs = []
    for lab in labels:
        if kept and lab == kept[-1]:
            evs.append(("retry", list(kept), lab, kept[-1]))
        elif lab in kept:
            kept.append(lab)
            evs.append(("back", list(kept), lab, kept[-2] if len(kept) >= 2 else None))
        else:
            kept.append(lab)
            evs.append(("advance", list(kept), lab, kept[-2] if len(kept) >= 2 else None))
    return evs


def lineage_selftest(rounds=200):
    rnd = random.Random(7)
    abc = "abcdefg"
    for _ in range(rounds):
        seq = [rnd.choice(abc) for _ in range(rnd.randint(3, 24))]
        ks_a = [e[0] for e in fold_events(seq)]
        ks_b = [ev["kind"] for ev in fold(seq)[1]]
        assert ks_a == ks_b, (seq, ks_a, ks_b)
    print("lineage self-test: fold_events == fold_rules.fold on 200 random sequences")


def build_corpus(name, rows, agent):
    train_fwd = defaultdict(Counter)
    for r in rows:
        if split_of(r["trace"]) != "train":
            continue
        kept = []
        for lab in r["labels"]:
            if kept and lab == kept[-1]:
                continue
            kept.append(lab)
        for a, b in zip(kept, kept[1:]):
            if a != b:
                train_fwd[a][b] += 1

    edge_vocab = {}
    for a, ctr in train_fwd.items():
        for b in ctr:
            edge_vocab[f"{a}->{b}"] = f"e{len(edge_vocab):04d}"
    actions = sorted({lab for r in rows for lab in r["labels"]})
    action_vocab = {a: i for i, a in enumerate(actions)}
    agent_vocab = {agent: 0}

    def features(kept_prefix, step_idx, n_steps):
        cur = kept_prefix[-1]
        vis = kept_prefix.count(cur)
        outs = train_fwd.get(cur, {})
        tot = sum(outs.values()) or 1
        top = max(outs.values()) / tot if outs else 0.0
        ent = -sum((c / tot) * math.log(c / tot + 1e-12) for c in outs.values()) if outs else 0.0
        return {"node": action_vocab[cur], "goal": 0, "step_idx": step_idx,
                "prefix_len": len(kept_prefix), "n_distinct": len(set(kept_prefix)),
                "revisit_ratio": 1.0 - len(set(kept_prefix)) / max(1, len(kept_prefix)),
                "visit_count": vis, "out_deg": len(outs), "out_topfreq": top,
                "out_entropy": ent}

    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        sp = split_of(r["trace"])
        evs = fold_events(r["labels"])
        stream = [(kind, kept_after, lab) for (kind, kept_after, lab, _p) in evs if kind != "retry"]
        n = len(stream)
        for i, (kind, kept_after, lab) in enumerate(stream):
            target = stream[i + 1][2] if i + 1 < n else STOP
            opts = sorted(train_fwd.get(kept_after[-1], {}).keys()) + [STOP]
            oel = ["<stop_edge>" if o == STOP else
                   edge_vocab.get(f"{kept_after[-1]}->{o}", "<unk_edge>") for o in opts]
            teid = ("<stop_edge>" if target == STOP else
                    edge_vocab.get(f"{kept_after[-1]}->{target}", "<unk_edge>"))
            splits[sp].append({
                "trace": r["trace"], "t": features(kept_after, i, n),
                "cur": kept_after[-1], "options": opts, "option_edge_ids": oel,
                "target": target, "target_edge_id": teid,
                "target_in_opts": target in opts, "kind": kind,
            })

    for sp, recs in splits.items():
        (PUB / f"{name}_e2_{sp}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    meta = {"action_vocab": action_vocab, "agent_vocab": agent_vocab,
            "edge_vocab": edge_vocab, "stop": STOP, "corpus": name}
    (PUB / f"{name}_e2_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    oos = sum(1 for r in splits["test"] if not r["target_in_opts"])
    print(f"{name:16s} decisions train/val/test = {len(splits['train'])}/{len(splits['val'])}/"
          f"{len(splits['test'])}  actions {len(action_vocab)}  edges {len(edge_vocab)}  "
          f"test out-of-space {100 * oos / max(1, len(splits['test'])):.1f}%")


lineage_selftest()
for p in sorted(PUB.glob("*_cases.jsonl")):
    name = p.name.replace("_cases.jsonl", "")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 50:
        print(f"{name:16s} skipped ({len(rows)} cases < 50)")
        continue
    build_corpus(name, rows, agent=name)