# E2 decision dataset from real folded episodes.
# Decision state = folded prefix; options = TRAIN-graph outgoing edges + stop;
# targets = next action class and next structural-edge class.
# Split by episode (sha256(trace)%%100): 0-69 train / 70-79 val / 80-99 test.
# Writes real/e2_train.jsonl / e2_val.jsonl / e2_test.jsonl + real/e2_meta.json (frozen).
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

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

# --- train graph: forward advance transitions (kept, skip repeats) ---------
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
        edge_vocab[(a, b)] = f"e{len(edge_vocab):04d}"

actions = sorted({lab for r in rows for lab in r["labels"]})
action_vocab = {a: i for i, a in enumerate(actions)}
agent_vocab = {}
for r in rows:
    agent_vocab.setdefault(r["agent"], len(agent_vocab))

def features(kept_prefix, agent, step_idx, n_steps):
    cur = kept_prefix[-1]
    vis = kept_prefix.count(cur)
    outs = train_fwd.get(cur, {})
    tot = sum(outs.values()) or 1
    top = max(outs.values()) / tot if outs else 0.0
    ent = -sum((c / tot) * math.log(c / tot + 1e-12) for c in outs.values()) if outs else 0.0
    return {
        "node": action_vocab[cur],
        "goal": agent_vocab[agent],
        "step_idx": step_idx,
        "prefix_len": len(kept_prefix),
        "n_distinct": len(set(kept_prefix)),
        "revisit_ratio": 1.0 - len(set(kept_prefix)) / max(1, len(kept_prefix)),
        "visit_count": vis,
        "out_deg": len(outs),
        "out_topfreq": top,
        "out_entropy": ent,
    }

splits = {"train": [], "val": [], "test": []}
stats = Counter()
for r in rows:
    sp = split_of(r["trace"])
    evs = fold_events(r["labels"])
    # decision stream = non-retry events (retries fold into loop events, §3)
    stream = [(kind, kept_after, lab) for (kind, kept_after, lab, _prev) in evs if kind != "retry"]
    n = len(stream)
    for i, (kind, kept_after, lab) in enumerate(stream):
        kept_prefix = kept_after
        target = stream[i + 1][2] if i + 1 < n else STOP
        opts = sorted(train_fwd.get(kept_prefix[-1], {}).keys())
        if not opts:
            opts = []
        opts = opts + [STOP]
        target_in_opts = target in opts
        # distinct classes: structural edge ids / <stop_edge> terminal / <unk_edge> unseen target
        options_edge_ids = ["<stop_edge>" if o == STOP else edge_vocab.get((kept_prefix[-1], o), "<unk_edge>")
                            for o in opts]
        teid = ("<stop_edge>" if target == STOP
                else edge_vocab.get((kept_prefix[-1], target), "<unk_edge>"))
        splits[sp].append({
            "trace": r["trace"],
            "t": features(kept_prefix, r["agent"], i, n),
            "cur": kept_prefix[-1],
            "options": opts,
            "option_edge_ids": options_edge_ids,
            "target": target,
            "target_edge_id": teid,
            "target_in_opts": target_in_opts,
            "kind": kind,
        })
        stats[f"{sp}_decisions"] += 1
        if not target_in_opts:
            stats[f"{sp}_target_outside"] += 1

meta = {
    "action_vocab": action_vocab,
    "agent_vocab": agent_vocab,
    "edge_vocab": {f"{a}->{b}": eid for (a, b), eid in edge_vocab.items()},
    "stop": STOP,
    "n_actions_total": len(actions) + 1,
    "n_edges_total": len(edge_vocab) + 1,
}
for sp, recs in splits.items():
    (OUT / f"e2_{sp}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
(OUT / "e2_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
print(json.dumps(dict(stats), indent=2))
print("vocab:", len(action_vocab), "actions +stop;", len(edge_vocab), "edges +stop-equiv")