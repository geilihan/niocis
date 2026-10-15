# ADDENDUM (item 3): E3 multi-seed -- removes the paper's "single-seed-0 refit chain" caveat.
#
# Reuses the CACHED mimo26 opponent stream (real/e3_mimo_stream.json, 3,875 decisions). Zero
# LLM calls: we hard-fail if that cache is absent rather than substituting a fresh request,
# so this run makes no network calls and cannot perturb the opponent.
#
# Recipe is verbatim e3_replay.py (probe = single Linear head on feats_laya_cls.npz, Adam
# 5e-3, 90 epochs, batch 512, RandomState(ep) shuffle; CE+Brier; masked-renorm over the
# declared options; the same teacher-forced match / strict survival / wrong-hop gap under
# c=1 and c=3), with exactly one knob opened: the torch init seed (0..4).
#
# SHAPE-0 SELF-CHECK: seed 0 must reproduce the frozen e3_replay.json numbers to the digit.
# Writes addendum/e3_replay_multi.json -- never touches real/.
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn

SRC = Path("/lw2/runs") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
OUT = Path("/lw2/addendum") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "addendum")
OUT.mkdir(exist_ok=True)
STOP = "<stop>"
N_SEEDS = 5

assert (SRC / "e3_mimo_stream.json").exists(), \
    "cached mimo stream missing -- refusing to call the LLM (this run is cache-only)"
MIMO_CACHE = SRC / "e3_mimo_stream.json"
mimo = {int(k): v for k, v in json.loads(MIMO_CACHE.read_text(encoding="utf-8")).items()}
print(f"mimo stream cached: {len(mimo)} decisions (0 LLM calls)", flush=True)

meta = json.loads((SRC / "e2_meta.json").read_text(encoding="utf-8"))
A2I = {**meta["action_vocab"], STOP: len(meta["action_vocab"])}
tr = [json.loads(l) for l in (SRC / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
te = [json.loads(l) for l in (SRC / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
Fe = np.load(SRC / "feats_laya_cls.npz")
ytr = np.array([A2I[r["target"]] for r in tr])


class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


def train_probe(seed):
    torch.manual_seed(seed)                       # <-- the only opened knob
    m = Probe(Fe["train"].shape[1], len(A2I))
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(Fe["train"].astype(np.float32))
    yt = torch.tensor(ytr, dtype=torch.long)
    for ep in range(90):
        order = np.random.RandomState(ep).permutation(len(ytr))
        for i in range(0, len(ytr), 512):
            b = order[i:i + 512]
            logits = m(Xt[b])
            p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            loss = ce + ((p - oh) ** 2).sum(1).mean()
            optz.zero_grad(); loss.backward(); optz.step()
    m.eval()
    logits = m(torch.tensor(Fe["test"].astype(np.float32))).detach().numpy()
    typed = []
    for i, rec in enumerate(te):
        opts = rec["options"]
        idx = [A2I[o] for o in opts]
        gl = logits[i][idx]; gl = gl - gl.max()
        p = np.exp(gl); p /= p.sum()
        typed.append({"pred": opts[int(p.argmax())], "conf": float(p.max())})
    del m, Xt, yt
    return typed


def evaluate(pred_fn, name):
    """verbatim from e3_replay.py -- teacher-forced match, strict survival, full wrong-hop gap."""
    by_trace = defaultdict(list)
    match, n = 0, len(te)
    for i, rec in enumerate(te):
        ok = (pred_fn(i) == rec["target"])
        match += ok
        by_trace[rec["trace"]].append((i, ok))
    surv, gap = 0, {"c1": 0.0, "c3": 0.0}
    n_eps = len(by_trace)
    for tid, decs in by_trace.items():
        dead = False
        for i, ok in decs:
            if not ok:
                dead = True
                gap["c1"] += 1.0
                gap["c3"] += 3.0
        surv += (not dead)
    return {"name": name, "decision_match": round(match / n, 4), "episodes": n_eps,
            "episode_survival": round(surv / n_eps, 4),
            "gap_per_episode": {"wrong_cost_1": round(gap["c1"] / n_eps, 4),
                                "wrong_cost_3": round(gap["c3"] / n_eps, 4)}}


if __name__ == "__main__":
    per_seed = []
    for s in range(N_SEEDS):
        typed = train_probe(s)
        r = {"seed": s, "typed_only": evaluate(lambda i, t=typed: t[i]["pred"], "typed_only")}
        for tau in (0.55, 0.75, 0.90):
            esc = sum(1 for t in typed if t["conf"] < tau)
            def tau_fn(i, tau=tau, t=typed):
                return mimo[i]["pred"] if t[i]["conf"] < tau else t[i]["pred"]
            q = evaluate(tau_fn, f"typed+esc@tau{tau}")
            q["escalation_rate"] = round(esc / len(te), 4)
            r[f"typed+esc@tau{tau}"] = q
        per_seed.append(r)
        print("seed", s, "typed", r["typed_only"]["decision_match"], "surv",
              r["typed_only"]["episode_survival"],
              "esc_rates", [r[f"typed+esc@tau{t}"]["escalation_rate"] for t in (0.55, 0.75, 0.90)],
              flush=True)

    frozen = {
        "typed_only": {"decision_match": 0.3066, "episode_survival": 0.0058},
    }
    try:
        fz = json.loads((SRC / "e3_replay.json").read_text(encoding="utf-8"))
        frozen = {k: {kk: fz[k][kk] for kk in ("decision_match", "episode_survival") if kk in fz[k]}
                  for k in ("mimo26_everyhop", "typed_only", "typed+esc@tau0.55",
                            "typed+esc@tau0.75", "typed+esc@tau0.9") if k in fz}
    except Exception as e:
        print("frozen read failed:", e)
    m0 = per_seed[0]
    def near(a, b):
        return a is not None and b is not None and abs(float(a) - float(b)) < 5e-4
    parity = True
    for k, v in frozen.items():
        if k in m0:
            for kk, vv in v.items():
                if kk in m0[k]:
                    parity &= near(m0[k][kk], vv)
    print("\nSHAPE-0 parity vs frozen e3_replay.json:", "PASS" if parity else "MISMATCH")

    def agg(key):
        idea = [p[key]["decision_match"] for p in per_seed]
        surv = [p[key]["episode_survival"] for p in per_seed]
        out = {"decision_match_mean": round(float(np.mean(idea)), 4),
               "decision_match_std": round(float(np.std(idea)), 4),
               "episode_survival_mean": round(float(np.mean(surv)), 4),
               "episode_survival_std": round(float(np.std(surv)), 4)}
        if "gap_per_episode" in per_seed[0][key]:
            for c in ("wrong_cost_1", "wrong_cost_3"):
                v = [p[key]["gap_per_episode"][c] for p in per_seed]
                out[f"gap_{c}_mean"] = round(float(np.mean(v)), 4)
                out[f"gap_{c}_std"] = round(float(np.std(v)), 4)
        if "escalation_rate" in per_seed[0][key]:
            v = [p[key]["escalation_rate"] for p in per_seed]
            out["escalation_rate_mean"] = round(float(np.mean(v)), 4)
            out["escalation_rate_std"] = round(float(np.std(v)), 4)
        return out

    keys = ["typed_only"] + [f"typed+esc@tau{t}" for t in (0.55, 0.75, 0.90)]
    res = {"_protocol": {
        "recipe": "verbatim e3_replay.py (Linear probe on feats_laya_cls.npz, Adam 5e-3, 90 epochs, batch 512, RandomState(ep) shuffle; masked-renorm over declared options)",
        "opened_knob": "torch init seed only (0..4)",
        "llm_calls": 0,
        "llm_source": "cached real/e3_mimo_stream.json -- refuse-to-run if absent",
        "self_check": "SHAPE-0 (seed 0) must match the frozen e3_replay.json digit-for-digit",
    }, "shape0_parity": parity,
        "frozen_reference": frozen, "aggregate": {k: agg(k) for k in keys}, "per_seed": per_seed}
    (OUT / "e3_replay_multi.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\nwrote", OUT / "e3_replay_multi.json")
    print(json.dumps(res["aggregate"], indent=1))