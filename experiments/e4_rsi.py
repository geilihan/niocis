# E4 on real data: gated self-evolution (RSI) under 12% label corruption.
# Base = hidden-state probe (action, CE+Brier, train-clean). Deployment stream = VAL decisions
# into 4 rounds of 300; round 3 injects 12% uniform label corruption. Ungated update absorbs
# poisoned labels into the refit; gated update is proposed-but-rolled-back if clean gate-set
# accuracy drops > 0.5pp. TEST stays clean and untouched by updates.
# Writes real/e4_rsi.json (frozen).
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
meta = json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8"))
STOP = meta["stop"]
action_vocab = meta["action_vocab"]
A2I = {**action_vocab, STOP: len(action_vocab)}
I2A = {v: k for k, v in A2I.items()}

tr = [json.loads(l) for l in (OUT / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
_sub = OUT / "train_sub_idx.json"
if _sub.exists():  # dormant: train_sub_idx.json is retired (file absent); the reported runs train on the full split
    tr = [tr[i] for i in json.loads(_sub.read_text(encoding="utf-8"))]
va = [json.loads(l) for l in (OUT / "e2_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
te = [json.loads(l) for l in (OUT / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
Fe = np.load(OUT / "feats_laya_cls.npz")

def y_of(recs):
    return np.array([A2I[r["target"]] for r in recs])

class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)

def fit(X, y, seed=0, epochs=60):
    torch.manual_seed(seed)
    m = Probe(X.shape[1], len(A2I))
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(X.astype(np.float32))
    yt = torch.tensor(y, dtype=torch.long)
    for ep in range(epochs):
        order = np.random.RandomState(ep).permutation(len(y))
        for i in range(0, len(y), 512):
            b = order[i:i + 512]
            logits = m(Xt[b])
            p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            loss = ce + ((p - oh) ** 2).sum(1).mean()
            optz.zero_grad(); loss.backward(); optz.step()
    return m

def acc_masked(m, recs, X):
    m.eval()
    logits = m(torch.tensor(X.astype(np.float32))).detach().numpy()
    ok = 0
    for i, r in enumerate(recs):
        idx = [A2I[o] for o in r["options"]]
        gl = logits[i][idx]
        j = int(np.exp(gl - gl.max()).argmax())
        ok += (idx[j] == A2I[r["target"]])
    return ok / len(recs)

# stream vs gate partitions of VAL (seeded permutation first: stream = first
# 1200 rows of the shuffled val, gate = the remaining 380 rows)
rng = np.random.RandomState(7)
perm = rng.permutation(len(va))
va_sh = [va[p] for p in perm]
Xv_sh = Fe["val"][perm]
stream, gate = va_sh[:1200], va_sh[1200:]
Xs, Xg = Xv_sh[:1200], Xv_sh[1200:]
y_gate = y_of(gate)

Xtr, ytr = Fe["train"], y_of(tr)
Xte, yte_te = Fe["test"], te  # kept clean

rounds = []
cur_train_X = [Xtr]
cur_train_y = [ytr]
model = fit(np.concatenate(cur_train_X), np.concatenate(cur_train_y))
gate_acc = acc_masked(model, gate, Xg)
test_acc = acc_masked(model, te, Xte)
rounds.append({"round": 0, "event": "base fit", "gate_acc": round(gate_acc, 4),
               "test_acc": round(test_acc, 4)})

B = 300
for r in range(1, 5):
    Xb = Xs[(r - 1) * B: r * B]
    recs_b = stream[(r - 1) * B: r * B]
    yb = y_of(recs_b)
    rate = 0.12 if r == 3 else 0.0
    applied = yb.copy()
    if rate > 0:
        rr = np.random.RandomState(100 + r)
        idx = rr.choice(len(yb), int(round(rate * len(yb))), replace=False)
        for i in idx:
            choices = [A2I[o] for o in recs_b[i]["options"] if A2I[o] != yb[i]]
            applied[i] = choices[rr.randint(len(choices))]

    # -- ungated: absorb update
    ung = fit(np.concatenate(cur_train_X + [Xb]), np.concatenate(cur_train_y + [applied]))
    ung_gate = acc_masked(ung, gate, Xg)
    ung_test = acc_masked(ung, te, Xte)
    rounds.append({"round": r, "event": "ungated update" + (f" +{int(rate*100)}% corrupt" if rate else ""),
                   "gate_acc": round(ung_gate, 4), "test_acc": round(ung_test, 4)})

    # -- gated: accept only if gate accuracy does not drop > 0.5pp
    if ung_gate >= gate_acc - 0.005:
        model = ung
        cur_train_X.append(Xb)
        cur_train_y.append(applied)
        gate_acc = ung_gate
        ev = "gated accepted"
    else:
        ev = "gated ROLLBACK"
    test_acc = acc_masked(model, te, Xte)
    rounds.append({"round": r, "event": ev, "gate_acc": round(gate_acc, 4),
                   "test_acc": round(test_acc, 4)})

(OUT / "e4_rsi.json").write_text(json.dumps(rounds, indent=2), encoding="utf-8")
print(json.dumps(rounds, indent=2))