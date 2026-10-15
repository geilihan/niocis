# E2 training on real decision dataset: action-class vs structural-edge IDs,
# CE vs CE+Brier proper scoring, 5 seeds, masked-renormalized gate + risk-coverage + ECE.
# Baselines: LUT (train-majority per node), HistGBM. Writes real/e2_results.json (frozen).
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
DEV = "cpu"
torch.manual_seed(0)

def load_split(name):
    return [json.loads(l) for l in (OUT / f"e2_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

tr, va, te = load_split("train"), load_split("val"), load_split("test")
meta = json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8"))
action_vocab = meta["action_vocab"]
agent_vocab = meta["agent_vocab"]
edge_vocab = meta["edge_vocab"]
STOP = meta["stop"]
A2I = {**action_vocab, STOP: len(action_vocab)}
# edge option/target IDs are "e####" strings (values of edge_vocab), not "a->b" keys
_edge_ids = sorted(set(edge_vocab.values()))
E2I = {eid: i for i, eid in enumerate(_edge_ids)}
E2I["<stop_edge>"] = len(E2I)
E2I["<unk_edge>"] = len(E2I) + 1
N_A = len(A2I)
N_E = len(E2I)
N_AGENT = len(agent_vocab)
NUMS = ["step_idx", "prefix_len", "n_distinct", "revisit_ratio", "visit_count",
        "out_deg", "out_topfreq", "out_entropy"]

def vec(rec, edge_target=False):
    t = rec["t"]
    x = np.zeros(len(A2I) + N_AGENT + len(NUMS) + 1, dtype=np.float32)
    x[t["node"]] = 1.0
    x[len(A2I) + t["goal"]] = 1.0
    base = len(A2I) + N_AGENT
    x[base + 0] = t["step_idx"] / 60.0
    x[base + 1] = t["prefix_len"] / 20.0
    x[base + 2] = t["n_distinct"] / 20.0
    x[base + 3] = t["revisit_ratio"]
    x[base + 4] = math.log1p(t["visit_count"]) / 3.0
    x[base + 5] = t["out_deg"] / 20.0
    x[base + 6] = t["out_topfreq"]
    x[base + 7] = t["out_entropy"] / 3.0
    x[base + 8] = 1.0 if t["first"] else 0.0
    return x

def build(recs, edge_target):
    X = np.stack([vec(r, edge_target) for r in recs])
    yi = [E2I.get(r["target_edge_id"], E2I["<unk_edge>"]) if edge_target else A2I[r["target"]]
          for r in recs]
    y = np.array(yi)
    opt_idx = []
    for r in recs:
        keys = r["option_edge_ids"] if edge_target else r["options"]
        vocab = E2I if edge_target else A2I
        unk = vocab["<unk_edge>"] if edge_target else A2I[STOP]
        opt_idx.append([vocab.get(k, unk) for k in keys])
    in_space = np.array([r["target_in_opts"] for r in recs])
    return X, y, opt_idx, in_space

# add first-step marker to features (legitimate: position-0 flag from the record itself)
for split in (tr, va, te):
    for r in split:
        r["t"]["first"] = (r["t"]["step_idx"] == 0)

class MLP(nn.Module):
    def __init__(self, d, k, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)

def masked_logits(logits, opt_idx):
    """Renormalize over offered options (the declared option space at the state)."""
    idx = torch.tensor(opt_idx, dtype=torch.long, device=logits.device)  # (B, M) ragged -> pad
    B, M = idx.shape
    g = logits.gather(1, idx)
    return g, idx

def train_model(Xtr, ytr, Xva, yva, opt_va, n_classes, objective, seed=0, epochs=60, edge_mode=False):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], n_classes).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        order = np.random.RandomState(seed + ep).permutation(n)
        for i in range(0, n, 512):
            b = order[i:i + 512]
            logits = model(Xt[b])
            p = torch.softmax(logits, dim=1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            brier = ((p - oh) ** 2).sum(dim=1).mean()
            loss = ce if objective == "ce" else ce + brier
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def eval_model(model, X, y, opt_idx, in_space):
    model.eval()
    logits = model(torch.tensor(X, device=DEV)).detach().cpu().numpy()
    n = len(y)
    correct = 0
    top1_unmasked = 0
    confs, cors = [], []
    preds = []
    rows = []
    for i in range(n):
        opts = opt_idx[i]
        gl = logits[i][opts]
        gl = gl - gl.max()
        p = np.exp(gl); p = p / p.sum()
        j = int(p.argmax())
        preds.append(opts[j])
        ok = (opts[j] == y[i])
        correct += ok
        top1_unmasked += (logits[i].argmax() == y[i])
        confs.append(float(p[j])); cors.append(ok)
        rows.append((float(p[j]), ok))
    rows.sort(key=lambda r: -r[0])
    errs_run = 0
    curve = []
    for k, (c, ok) in enumerate(rows, 1):
        errs_run += (not ok)
        if k in (n // 20, n // 10, n // 4, n // 2, n) or k % max(1, n // 100) == 0:
            curve.append({"coverage": round(k / n, 4), "risk": round(errs_run / k, 5)})
    # ECE 15 bins
    confs_a, cors_a = np.array(confs), np.array(cors, dtype=float)
    ece = 0.0
    for lo in range(15):
        a, b = lo / 15, (lo + 1) / 15
        m = (confs_a > lo / 15) & (confs_a <= b) if lo else (confs_a >= 0) & (confs_a <= b)
        if m.sum():
            ece += m.mean() * abs(confs_a[m].mean() - cors_a[m].mean())
    # risk at high coverage operating points
    def risk_at(cov):
        k = max(1, int(round(cov * n)))
        return round(sum(1 for c, ok in rows[:k] if not ok) / k, 5)
    # AURC
    aurc = sum((sum(1 for c2, ok2 in rows[:k] if not ok2) / k) / n for k in range(1, n + 1))
    in_space = np.asarray(in_space, dtype=bool)
    cors_a_is = np.array([opts_ok for opts_ok in cors])[in_space] if in_space.any() else np.array([])
    return {
        "masked_acc": round(correct / n, 4),
        "acc_declared": round(float(cors_a_is.mean()), 4) if in_space.any() else None,
        "n_in_space": int(in_space.sum()),
        "unmasked_acc": round(top1_unmasked / n, 4),
        "ece15": round(float(ece), 4),
        "aurc": round(aurc, 4),
        "risk_at_cov": {str(c): risk_at(c) for c in (0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0)},
        "curve": curve,
    }

results = {}

for edge_mode, tag, n_classes in ((False, "action", N_A), (True, "edge", N_E)):
    Xtr, ytr, _, _ = build(tr, edge_mode)
    Xva, yva, opt_va, isp_va = build(va, edge_mode)
    Xte, yte, opt_te, isp_te = build(te, edge_mode)

    # LUT: per current label, majority train target
    from collections import Counter, defaultdict
    ml = defaultdict(Counter)
    for r, yy in zip(tr, ytr):
        ml[r["cur"]][yy] += 1
    for cand, nm in (("lut", 0),):
        okv = 0
        for r, yy in zip(va, yva):
            pred = ml.get(r["cur"], Counter()).most_common(1)[0][0] if ml.get(r["cur"]) else ytr[0]
            okv += (pred == yy)
        okt = 0
        for r, yy in zip(te, yte):
            pred = ml.get(r["cur"], Counter()).most_common(1)[0][0] if ml.get(r["cur"]) else ytr[0]
            okt += (pred == yy)
        results[f"{tag}_lut_majority"] = {"val_acc": round(okv / len(yva), 4), "test_acc": round(okt / len(yte), 4)}

    # Linear (multinomial logreg) baseline — deterministic and fast at both 155 and
    # 2,140 classes. (Tree ensembles are OvR per class and were infeasible on this host;
    # the paper's real-data table names this row explicitly.)
    lr = LogisticRegression(max_iter=800, random_state=0).fit(Xtr, ytr)
    results[f"{tag}_logreg"] = {
        "val_acc": round(float((lr.predict(Xva) == yva).mean()), 4),
        "test_acc": round(float((lr.predict(Xte) == yte).mean()), 4),
    }

    for objective in ("ce", "cebrier"):
        seed_stats = []
        for seed in range(5):
            m = train_model(Xtr, ytr, Xva, yva, opt_va, n_classes, objective, seed=seed)
            ev = eval_model(m, Xte, yte, opt_te, isp_te)
            seed_stats.append(ev)
        keys = ("masked_acc", "acc_declared", "unmasked_acc", "ece15", "aurc")
        agg = {}
        for k in keys:
            vs = [s[k] for s in seed_stats]
            agg[k] = {"mean": round(float(np.mean(vs)), 4), "std": round(float(np.std(vs)), 4)}
        agg["risk_at_cov_mean"] = {
            c: round(float(np.mean([s["risk_at_cov"][c] for s in seed_stats])), 5) for c in seed_stats[0]["risk_at_cov"]
        }
        agg["per_seed"] = [{k: s[k] for k in keys} for s in seed_stats]
        agg["curve_mean"] = seed_stats[0]["curve"]
        results[f"{tag}_mlp_{objective}"] = agg

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("curve", "per_seed", "curve_mean")} for k, v in results.items()}, indent=2))
(OUT / "e2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved ->", OUT / "e2_results.json")