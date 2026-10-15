# Slices & error bars companion: retrain the hidden-state probe (action, CE+Brier) per seed and
# report test accuracy by domain (agent) slice and by node out-degree bucket (local branch factor).
# Writes real/e8_slices.json (frozen).
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
meta = json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8"))
STOP = meta["stop"]
A2I = {**meta["action_vocab"], STOP: len(meta["action_vocab"])}

tr = [json.loads(l) for l in (OUT / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
_sub = OUT / "train_sub_idx.json"
if _sub.exists():  # dormant: train_sub_idx.json is retired (file absent); the reported runs train on the full split
    tr = [tr[i] for i in json.loads(_sub.read_text(encoding="utf-8"))]
te = [json.loads(l) for l in (OUT / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
trace_agent = {r["trace"]: r["agent"] for r in rows}
Fe = np.load(OUT / "feats_laya_cls.npz")

class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)

def fit(X, y, seed):
    torch.manual_seed(seed)
    m = Probe(X.shape[1], len(A2I))
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(X.astype(np.float32))
    yt = torch.tensor(y, dtype=torch.long)
    for ep in range(60):
        order = np.random.RandomState(seed + ep).permutation(len(y))
        for i in range(0, len(y), 512):
            b = order[i:i + 512]
            logits = m(Xt[b])
            p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            loss = ce + ((p - oh) ** 2).sum(1).mean()
            optz.zero_grad(); loss.backward(); optz.step()
    return m

ytr = np.array([A2I[r["target"]] for r in tr])
Xte = Fe["test"]

def bucket(rec):
    d = rec["t"]["out_deg"]
    return "<=3" if d <= 3 else ("4-10" if d <= 10 else "11+")

slices = {"by_agent": {}, "by_outdeg": {}}
per_seed_total = []
for seed in range(5):
    m = fit(Fe["train"], ytr, seed)
    m.eval()
    logits = m(torch.tensor(Xte.astype(np.float32))).detach().numpy()
    hits = []
    for i, r in enumerate(te):
        idx = [A2I[o] for o in r["options"]]
        gl = logits[i][idx]
        j = int(np.exp(gl - gl.max()).argmax())
        hits.append((trace_agent[r["trace"]], bucket(r), idx[j] == A2I[r["target"]]))
    per_seed_total.append(sum(h[2] for h in hits) / len(hits))
    for ag, bk, ok in hits:
        slices["by_agent"].setdefault(ag, []).append(ok)
        slices["by_outdeg"].setdefault(bk, []).append(ok)

def agg(vals):
    return {"n": len(vals), "acc": round(float(np.mean(vals)), 4)}

result = {
    "test_acc_per_seed": [round(v, 4) for v in per_seed_total],
    "test_acc_mean": round(float(np.mean(per_seed_total)), 4),
    "test_acc_std": round(float(np.std(per_seed_total)), 4),
    "by_agent": {k: agg(v) for k, v in slices["by_agent"].items()},
    "by_outdeg": {k: agg(v) for k, v in sorted(slices["by_outdeg"].items())},
}
(OUT / "e8_slices.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))