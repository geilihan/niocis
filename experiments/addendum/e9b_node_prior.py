# E9b: sharpen E9 -- does REACHABILITY add anything beyond NODE-CONDITIONAL frequency?
#
# E9 showed the declared A(v) beats a size-matched GLOBAL frequency prior by 1.37 pts.
# The natural next competitor uses the SAME node conditioning as A(v) but no graph traversal:
#   topK_node : the K=|declared| most frequent TRAIN targets observed AT THE SAME NODE (cur)
#   union_node: every target observed at that node anywhere in TRAIN (no size matching)
# If topK_node closes the gap, the paper's claim must narrow from "the induced graph" to
# "node-conditioned option sets". If it does not, reachability itself carries the gain.
#
# Same training as E9 (verbatim published probe recipe); only eval option sets change.
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path("/lw2/runs") if Path("/lw2").exists() else Path(r"D:\ml\论文2\experiments\real")
OUT = Path("/lw2/addendum") if Path("/lw2").exists() else Path(r"D:\ml\论文2\experiments\addendum")

meta = json.loads((SRC / "e2_meta.json").read_text(encoding="utf-8"))
A2I = {**meta["action_vocab"], meta["stop"]: len(meta["action_vocab"])}
K_CLASS = len(A2I)
tr = [json.loads(l) for l in (SRC / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
te = [json.loads(l) for l in (SRC / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
F = np.load(SRC / "feats_laya_cls.npz")
print("cur examples:", [r["cur"] for r in te[:5]], flush=True)
print("t examples:", [str(r["t"])[:120] for r in te[:2]], flush=True)

y_tr = np.array([A2I[r["target"]] for r in tr])
y_te = np.array([A2I[r["target"]] for r in te])
declared = [[A2I[o] for o in r["options"]] for r in te]
sizes = np.array([len(o) for o in declared])

# node-conditional train statistics (TRAIN ONLY -> no test leakage)
node_freq = defaultdict(Counter)
for r in tr:
    node_freq[str(r["t"]["node"])][A2I[r["target"]]] += 1
n_nodes = len(node_freq)
covered = sum(1 for r in te if r["cur"] in node_freq)
print(f"nodes in train: {n_nodes} | test rows whose node is known: {covered}/{len(te)}", flush=True)


def node_set(rec, k=None):
    c = node_freq.get(rec["cur"])
    if not c:
        return list(range(K_CLASS))
    ranked = [a for a, _ in c.most_common()]
    return ranked[:k] if k else ranked


def options(arm):
    if arm == "declared":
        return declared
    if arm == "topK_node":
        return [(node_set(r, int(k)) or list(range(K_CLASS))) for r, k in zip(te, sizes)]
    if arm == "union_node":
        return [node_set(r) for r in te]
    if arm == "flat155":
        return [list(range(K_CLASS))] * len(te)
    raise ValueError(arm)


class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = ["declared", "topK_node", "union_node", "flat155"]
OPTS = {a: options(a) for a in ARMS}
print("mean |options| per arm:", {a: round(float(np.mean([len(o) for o in OPTS[a]])), 2) for a in ARMS}, flush=True)


def fit(seed, epochs=90):
    torch.manual_seed(seed)
    Xtr = F["train"].astype(np.float32)
    m = Probe(Xtr.shape[1], K_CLASS).to(DEV)
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEV)
    for ep in range(epochs):
        order = np.random.RandomState(seed + ep).permutation(len(y_tr))
        for i in range(0, len(y_tr), 512):
            b = order[i:i + 512]
            logits = m(Xt[b]); p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            loss = ce + ((p - oh) ** 2).sum(1).mean()
            optz.zero_grad(); loss.backward(); optz.step()
    m.eval()
    with torch.inference_mode():
        lg = m(torch.tensor(F["test"].astype(np.float32), device=DEV)).cpu().numpy()
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return lg


def correct(logits, opts):
    out = np.zeros(len(y_te), dtype=bool)
    for i, o in enumerate(opts):
        gl = logits[i][o]; gl = gl - gl.max()
        out[i] = (o[int(np.argmax(np.exp(gl)))] == y_te[i])
    return out


def cluster_ci(delta, traces, nb=2000):
    rng = np.random.RandomState(0)
    keys = np.array(sorted(set(traces)))
    idx = {k: np.where(traces == k)[0] for k in keys}
    m = np.empty(nb)
    for b in range(nb):
        pick = rng.choice(len(keys), len(keys), replace=True)
        m[b] = delta[np.concatenate([idx[keys[p]] for p in pick])].mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


traces = np.array([str(r["trace"]) for r in te])
in_space = np.array([bool(r.get("target_in_opts", True)) for r in te])
runs = []
for seed in range(5):
    t0 = time.perf_counter()
    lg = fit(seed)
    rec = {"seed": seed, "arms": {}}
    for a in ARMS:
        cm = correct(lg, OPTS[a])
        rec["arms"][a] = {"acc_all": round(float(cm.mean()), 4), "acc_in_space": round(float(cm[in_space].mean()), 4)}
    runs.append(rec)
    print(f"seed{seed} " + " ".join(f"{a}={rec['arms'][a]['acc_all']:.4f}" for a in ARMS) + f" ({time.perf_counter()-t0:.1f}s)", flush=True)

lg0 = fit(0)
base = correct(lg0, OPTS["declared"])
agg = {}
for a in ARMS:
    vs = np.array([r["arms"][a]["acc_all"] for r in runs])
    agg[a] = {"mean": round(float(vs.mean()), 4), "std": round(float(vs.std()), 4),
              "acc_in_space_mean": round(float(np.mean([r["arms"][a]["acc_in_space"] for r in runs])), 4)}
for a in ARMS[1:]:
    d = correct(lg0, OPTS[a]).astype(float) - base.astype(float)
    lo, hi = cluster_ci(d, traces)
    agg[a]["paired_delta_vs_declared_seed0"] = round(float(d.mean()), 4)
    agg[a]["paired_delta_ci95_cluster"] = [round(lo, 4), round(hi, 4)]
    agg[a]["sign_test"] = {"wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}

res = {"protocol": "E9b: node-conditional graph-free competitors vs declared A(v); same training recipe as E9/e2_laya_probe",
       "nodes_in_train": n_nodes, "test_rows_node_known": covered, "n_test": len(te),
       "mean_options_per_arm": {a: round(float(np.mean([len(o) for o in OPTS[a]])), 2) for a in ARMS},
       "ci": "paired cluster bootstrap over traces, 2000 resamples (seed-0 model)", "runs": runs, "aggregate": agg}
(OUT / "e9b_node_prior_nodekey.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("\nAGGREGATE", json.dumps(agg, indent=1), flush=True)
print("wrote", OUT / "e9b_node_prior_nodekey.json")
