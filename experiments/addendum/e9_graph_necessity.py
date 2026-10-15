# E9: graph-necessity ablation (answers panel B-1 / R1-W6 / R3-W4 / DA-C2).
#
# Question: the probe arm reads ONLY the frozen Laya hidden state; the induced graph enters
# ONLY through the declared-option mask. So "is the graph necessary?" reduces, on this arm,
# to: does the declared option space A(v) beat a SIZE-MATCHED graph-free option space, at the
# identical readout, features and budget?
#
# Design (fully controlled):
#   * training is IDENTICAL in every arm and verbatim the published recipe
#     (e2_laya_probe.py: Probe(1024,k=155), Adam 5e-3, 90 epochs, batch 512,
#      RandomState(seed+ep) shuffle, objective CE or CE+Brier; NO masking in training).
#   * only the EVALUATION option space changes:
#       declared  : r["options"]                      (published arm -> parity receipt)
#       flat155   : all 155 action classes            (the graph-free variant named at :1085)
#       topK_freq : K = |declared| most frequent TRAIN targets   (size-matched, graph-free)
#       randK     : K = |declared| random classes                (floor)
#   * uncertainty: CLUSTER bootstrap over traces/episodes (paired per row), 2000 resamples,
#     because decisions inside an episode are not independent.
#
# Reads  /lw2/runs/{feats_laya_cls.npz,e2_{train,val,test}.jsonl,e2_meta.json,e2_layaprobe_results.json}
# Writes /lw2/addendum/e9_graph_necessity.json
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path("/lw2/runs") if Path("/lw2").exists() else Path(r"D:\ml\论文2\experiments\real")
OUT = Path("/lw2/addendum") if Path("/lw2").exists() else Path(r"D:\ml\论文2\experiments\addendum")
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((SRC / "e2_meta.json").read_text(encoding="utf-8"))
A2I = {**meta["action_vocab"], meta["stop"]: len(meta["action_vocab"])}
K_CLASS = len(A2I)
tr = [json.loads(l) for l in (SRC / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
te = [json.loads(l) for l in (SRC / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
F = np.load(SRC / "feats_laya_cls.npz")
print("train", len(tr), "test", len(te), "classes", K_CLASS, "feat", F["train"].shape, flush=True)
print("record keys:", sorted(te[0].keys()), flush=True)

y_tr = np.array([A2I[r["target"]] for r in tr])
y_te = np.array([A2I[r["target"]] for r in te])
declared = [[A2I[o] for o in r["options"]] for r in te]
sizes = np.array([len(o) for o in declared])
print("declared size: mean %.2f median %d min %d max %d" % (sizes.mean(), np.median(sizes), sizes.min(), sizes.max()), flush=True)

# graph-free size-matched priors
freq_order = np.argsort(-np.bincount(y_tr, minlength=K_CLASS), kind="stable")  # train target frequency
rng_rand = np.random.RandomState(20261012)
rand_sets = {k: rng_rand.choice(K_CLASS, size=int(k), replace=False).tolist() for k in np.unique(sizes)}


def option_sets(arm):
    if arm == "declared":
        return declared
    if arm == "flat155":
        return [list(range(K_CLASS))] * len(te)
    if arm == "topK_freq":
        return [freq_order[:int(k)].tolist() for k in sizes]
    if arm == "randK":
        return [rand_sets[int(k)] for k in sizes]
    raise ValueError(arm)


class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fit(seed, objective, epochs=90):
    """verbatim e2_laya_probe.py run(): Adam 5e-3, 90 epochs, batch 512, RandomState(seed+ep)."""
    torch.manual_seed(seed)
    Xtr = F["train"].astype(np.float32)
    m = Probe(Xtr.shape[1], K_CLASS).to(DEV)
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(y_tr, dtype=torch.long, device=DEV)
    n = len(y_tr)
    for ep in range(epochs):
        order = np.random.RandomState(seed + ep).permutation(n)
        for i in range(0, n, 512):
            b = order[i:i + 512]
            logits = m(Xt[b])
            p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            brier = ((p - oh) ** 2).sum(1).mean()
            loss = ce if objective == "ce" else ce + brier
            optz.zero_grad(); loss.backward(); optz.step()
    m.eval()
    with torch.inference_mode():
        lg = m(torch.tensor(F["test"].astype(np.float32), device=DEV)).cpu().numpy()
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return lg


def correct_mask(logits, opts_list, y):
    """per-row correctness under masked-renormalised argmax (published eval)"""
    out = np.zeros(len(y), dtype=bool)
    for i, opts in enumerate(opts_list):
        gl = logits[i][opts]
        gl = gl - gl.max()
        p = np.exp(gl)
        j = int(np.argmax(p))
        out[i] = (opts[j] == y[i])
    return out


def cluster_ci(delta, traces, n_boot=2000, seed=0):
    """paired cluster bootstrap over traces -> 95% CI of the mean delta."""
    rng = np.random.RandomState(seed)
    keys = np.array(sorted(set(traces)))
    idx = {k: np.where(traces == k)[0] for k in keys}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sel = np.concatenate([idx[keys[p]] for p in pick])
        means[b] = delta[sel].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


traces = np.array([str(r["trace"]) for r in te])
in_space = np.array([bool(r.get("target_in_opts", True)) for r in te])
ARMS = ["declared", "flat155", "topK_freq", "randK"]
OPTS = {a: option_sets(a) for a in ARMS}

runs = []
for objective in ("cebrier", "ce"):
    for seed in range(5):
        t0 = time.perf_counter()
        lg = fit(seed, objective)
        rec = {"objective": objective, "seed": seed, "arms": {}}
        for a in ARMS:
            cm = correct_mask(lg, OPTS[a], y_te)
            rec["arms"][a] = {
                "acc_all": round(float(cm.mean()), 4),
                "acc_in_space": round(float(cm[in_space].mean()), 4),
            }
        for a in ARMS[1:]:
            rec["arms"][a]["delta_vs_declared_all"] = round(rec["arms"][a]["acc_all"] - rec["arms"]["declared"]["acc_all"], 4)
        rec["secs"] = round(time.perf_counter() - t0, 1)
        runs.append(rec)
        print(f"{objective} seed{seed} " + " ".join(f"{a}={rec['arms'][a]['acc_all']:.4f}" for a in ARMS) +
              f"  ({rec['secs']}s)", flush=True)

# ---- aggregate + cluster CIs (seed 0 model of each objective, all seeds pooled for point est.) ----
summary = {"protocol": {
    "train": "verbatim e2_laya_probe.py run(): Probe(1024,155), Adam 5e-3, 90 epochs, batch 512, RandomState(seed+ep), no training mask",
    "eval_arms": {
        "declared": "r['options'] (published)",
        "flat155": "all 155 classes (graph-free variant named at main_body.tex:1085)",
        "topK_freq": "K=|declared| most frequent TRAIN targets (size-matched, graph-free)",
        "randK": "K=|declared| random classes (floor)",
    },
    "ci": "paired cluster bootstrap over traces, 2000 resamples",
}, "n_test": len(te), "n_traces": int(len(set(traces))), "declared_size_mean": round(float(sizes.mean()), 3), "runs": runs}

agg = {}
for objective in ("cebrier", "ce"):
    sel = [r for r in runs if r["objective"] == objective]
    agg[objective] = {}
    for a in ARMS:
        vs = np.array([r["arms"][a]["acc_all"] for r in sel])
        agg[objective][a] = {"mean": round(float(vs.mean()), 4), "std": round(float(vs.std()), 4)}
    # CIs from the first seed's model (paired, cluster bootstrap) -- stated as seed-0 model
    seed0 = sel[0]["seed"]
    lg = fit(seed0, objective)
    base = correct_mask(lg, OPTS["declared"], y_te)
    for a in ARMS[1:]:
        cm = correct_mask(lg, OPTS[a], y_te)
        d = cm.astype(float) - base.astype(float)
        lo, hi = cluster_ci(d, traces)
        agg[objective][a]["paired_delta_vs_declared_mean"] = round(float(d.mean()), 4)
        agg[objective][a]["paired_delta_ci95_cluster"] = [round(lo, 4), round(hi, 4)]
        agg[objective][a]["sign_test"] = {"wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}
summary["aggregate"] = agg

# ---- parity receipt vs the frozen published probe ----
try:
    frozen = json.loads((SRC / "e2_layaprobe_results.json").read_text(encoding="utf-8"))
    pub = frozen["action_layaprobe_cebrier"]["masked_acc"]
    mine = agg["cebrier"]["declared"]
    summary["parity_vs_published"] = {
        "published_masked_acc": pub, "mine": mine,
        "abs_diff": round(abs(pub["mean"] - mine["mean"]), 4),
        "tolerance": 0.005,
        "PASS": bool(abs(pub["mean"] - mine["mean"]) <= 0.005),
    }
    print("\nPARITY vs published action probe (cebrier masked acc):", pub, "vs", mine, flush=True)
except Exception as e:  # noqa: BLE001
    summary["parity_vs_published"] = {"error": f"{type(e).__name__}: {e}"}

(OUT / "e9_graph_necessity.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("\nAGGREGATE")
for objective in ("cebrier", "ce"):
    print(" ", objective, {a: agg[objective][a]["mean"] for a in ARMS}, flush=True)
    for a in ARMS[1:]:
        print("    ", a, "delta", agg[objective][a]["paired_delta_vs_declared_mean"],
              "CI", agg[objective][a]["paired_delta_ci95_cluster"], agg[objective][a]["sign_test"], flush=True)
print("wrote", OUT / "e9_graph_necessity.json")
