# ADDENDUM (item 2a): does the unused result-side telemetry actually widen the readout?
#
# The limitation ("state reconstruction": tool-result bodies and chat bodies are absent) is
# TRUE for the payloads, but two result-adjacent families survive in the export and are
# unused: per-call `latency` and the call `input` arguments. Both are already extracted into
# real/episodes.jsonl as `latencies[]` and `inputs[]` (indexed by raw call, aligned with
# `events[]` which records the fold decision per call).
#
# Design (same recipe as the published feature-MLP arm):
#   X_base = node one-hot (t['node']) + goal one-hot (t['goal']) + the 8 numeric stats
#            (step_idx, prefix_len, n_distinct, revisit_ratio, visit_count, out_deg,
#             out_topfreq, out_entropy) + a first-flag                    [SEC text]
#   X_aug  = X_base + prefix latency stats + per-call argument-shape counts
#   arm    = 2-layer MLP (128), CE + Brier, 5 seeds, masked-renorm over declared options
#
# SHAPE-0 SELF-CHECK: X_base must reproduce the published feature-MLP masked accuracy
# (32.8% = 0.3278 mean over seeds) to within 0.2 points before ANY uplift number is trusted.
# If it does not, we report PARITY FAIL and the uplift is not a result.
#
# Reads real/ + the raw-derived episodes.jsonl; WRITES addendum/feat_lat_results.json only.
import json, math
from pathlib import Path

import numpy as np
import torch, torch.nn as nn

SRC = Path("/lw2/runs") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
OUT = Path("/lw2/addendum") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "addendum")
OUT.mkdir(exist_ok=True)

SEEDS, EPOCHS, BATCH, HID = 5, 60, 512, 128
NUMS = ["step_idx", "prefix_len", "n_distinct", "revisit_ratio", "visit_count",
        "out_deg", "out_topfreq", "out_entropy"]

meta = json.loads((SRC / "e2_meta.json").read_text(encoding="utf-8"))
STOP = meta["stop"]
A2I = {**meta["action_vocab"], STOP: len(meta["action_vocab"])}
NGOAL = 1 + max(json.loads(l).get("t", {}).get("goal", 0)
                for l in (SRC / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())

rows = {sp: [json.loads(l) for l in (SRC / f"e2_{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for sp in ("train", "val", "test")}
eps = {e["trace"]: e for e in
       (json.loads(l) for l in (SRC / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

IDS = sorted({tuple(sorted(i.keys())) for e in eps.values() for i in e.get("inputs", []) if i})
IMAP = {k: j for j, k in enumerate(IDS)}
NF_LAT, NF_ARG, NF_BASE = 7, len(IMAP), len(A2I) + NGOAL + len(NUMS) + 1
print("arg-shape buckets:", {str(k): v for k, v in IMAP.items()}, flush=True)


def kept_idx(e):
    """fold decision per raw call: keep every call whose event is not a retry."""
    ev = e["events"]
    return [i for i, x in enumerate(ev) if x != "retry"]


def build(sp):
    rs = rows[sp]
    B = np.zeros((len(rs), NF_BASE), dtype=np.float32)
    L = np.zeros((len(rs), NF_LAT), dtype=np.float32)
    G = np.zeros((len(rs), NF_ARG), dtype=np.float32)
    y = np.array([A2I[r["target"]] for r in rs], dtype=np.int64)
    opts = [[A2I[o] for o in r["options"]] for r in rs]
    for i, r in enumerate(rs):
        t = r["t"]
        B[i, t["node"]] = 1.0
        B[i, len(A2I) + t.get("goal", 0)] = 1.0
        b = len(A2I) + NGOAL
        B[i, b:b + len(NUMS)] = [t["step_idx"] / 60.0, t["prefix_len"] / 20.0, t["n_distinct"] / 20.0,
                                 t["revisit_ratio"], math.log1p(t["visit_count"]) / 3.0,
                                 t["out_deg"] / 20.0, t["out_topfreq"], t["out_entropy"] / 3.0]
        B[i, -1] = 1.0 if t.get("first", t["step_idx"] == 0) else 0.0
        e = eps.get(r["trace"])
        if not e:
            continue
        ki = kept_idx(e)
        k = min(t["step_idx"], len(ki) - 1)
        seq = ki[:k + 1]                                  # calls observed up to the current state
        lat = [e["latencies"][j] for j in seq if j < len(e.get("latencies", []))]
        if lat:
            L[i] = [lat[-1], float(np.mean(lat)), float(np.max(lat)), float(np.std(lat)),
                    float(np.sum(lat)), float(len(lat)),
                    (lat[-1] - float(np.mean(lat[:-1]))) if len(lat) > 1 else 0.0]
        for j in seq:
            if j < len(e.get("inputs", [])):
                G[i, IMAP.get(tuple(sorted((e["inputs"][j] or {}).keys())), 0)] += 1.0
    # log-scale the latency family so the LR is well conditioned
    L[:, :5] = np.log1p(L[:, :5])
    G = np.log1p(G)
    return B, L, G, y, opts


DATA = {sp: build(sp) for sp in ("train", "test")}
print("built", {s: (DATA[s][0].shape, DATA[s][3].shape[0]) for s in DATA}, flush=True)


class MLP(nn.Module):
    def __init__(self, d, k, hidden=HID):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def run(feats):
    Xtr = np.hstack([g for g in feats[0]])
    Xte = np.hstack([g for g in feats[1]])
    ytr, opts = DATA["train"][3], DATA["train"][4]
    yte, ote = DATA["test"][3], DATA["test"][4]
    hits = []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        m = MLP(Xtr.shape[1], len(A2I))
        o = torch.optim.Adam(m.parameters(), lr=2e-3)
        Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
        for ep in range(EPOCHS):
            order = np.random.RandomState(seed + ep).permutation(len(ytr))
            for i in range(0, len(ytr), BATCH):
                b = order[i:i + BATCH]
                lg = m(Xt[b]); p = torch.softmax(lg, 1)
                ce = nn.functional.cross_entropy(lg, yt[b])
                oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
                loss = ce + ((p - oh) ** 2).sum(1).mean()
                o.zero_grad(); loss.backward(); o.step()
        m.eval()
        with torch.no_grad():
            Lo = m(torch.tensor(Xte)).detach().numpy()
        ok = 0
        for i in range(len(yte)):
            idx = ote[i]; g = Lo[i][idx]; g = g - g.max()
            ok += (idx[int(np.exp(g).argmax())] == yte[i])
        hits.append(ok / len(yte))
        del m, Xt, yt
    return float(np.mean(hits)), float(np.std(hits)), [round(h, 4) for h in hits]


if __name__ == "__main__":
    (BBtr, LLtr, GGtr, _, _), (BBte, LLte, GGte, _, _) = DATA["train"], DATA["test"]
    base = run((((BBtr,), (BBte,))))
    aug_lat = run(((BBtr, LLtr), (BBte, LLte)))
    aug_all = run(((BBtr, LLtr, GGtr), (BBte, LLte, GGte)))
    SHAKED = 32.78 / 100.0
    parity = abs(base[0] - SHAKED) <= 0.002
    print("\nSHAPE-0 baseline parity:", "PASS" if parity else "FAIL",
          f"(base {base[0]*100:.2f}% vs published 32.78%)", flush=True)
    res = {"_protocol": {
        "recipe": "verbatim feature-MLP arm (2x128 ReLU MLP, CE+Brier, masked-renorm over declared options)",
        "X_base": "node one-hot (t['node']) + goal one-hot + 8 numeric stats + first-flag",
        "X_lat": "prefix latency {last, mean, max, std, sum, count, trend} (log1p)",
        "X_arg": "per-prefix call-input key-shape counts (log1p)",
        "n_seeds": SEEDS, "epochs": EPOCHS, "batch": BATCH,
        "self_check": "X_base must reproduce the published 32.78% within 0.2 points",
    }, "shape0_parity": parity,
        "baseline_masked": {"mean": round(base[0], 4), "std": round(base[1], 4), "per_seed": base[2]},
        "plus_latency": {"mean": round(aug_lat[0], 4), "std": round(aug_lat[1], 4), "per_seed": aug_lat[2]},
        "plus_lat_arg": {"mean": round(aug_all[0], 4), "std": round(aug_all[1], 4), "per_seed": aug_all[2]},
        "delta_latency_pts": round((aug_lat[0] - base[0]) * 100, 3),
        "delta_lat_arg_pts": round((aug_all[0] - base[0]) * 100, 3),
    }
    (OUT / "feat_lat_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "_protocol"}, indent=1))
    print("wrote", OUT / "feat_lat_results.json")