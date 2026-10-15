# E3 real held-out replay: mimo26 per-hop opponent + typed readout + gated hybrid.
# One mimo26 call per TEST decision serves every LLM-dependent arm (opponent, hybrid at taus).
# Teacher-forced scoring (decision match) + strict survival (episode ends at first mismatch)
# + gap under two cost conventions ({1,3} x wrong hop, counted over the whole episode,
#   teacher-forced - not truncated at first divergence) + real measured latency.
# Writes real/e3_replay.json (frozen).
import os  # R-1
import json
import math
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

OUT = Path("/lw2/runs") if Path("/lw2/runs").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
BASE = os.environ.get("ARIADNE_LLM_BASE",
       "https://<your-gateway-host>/v1/chat/completions")  # R-1: endpoint from env (SECURITY-NOTES.md)
KEY = os.environ.get("ARIADNE_LLM_KEY", "")  # R-1: rotate the leaked key; never commit one
GOALS = {
    "detection": "detect-stage: gather log evidence for the alert and judge triage disposition",
    "protection": "protect-stage: execute protective handling and record summary",
    "response": "response-stage: investigate attack path and act, record response report",
    "recognition": "recognize-stage: profile entities and record recognition report",
    "dlp-dispose": "dispose-stage: handle DLP alert and record disposal summary",
    "QwenPaw_QA_Agent_0.2": "answer the user question",
    "unknown": "complete the current task",
}
STOP = "<stop>"

te = [json.loads(l) for l in (OUT / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
trace_ep = {r["trace"]: r for r in rows}

def state_text(rec):
    ep = trace_ep[rec["trace"]]
    goal = GOALS.get(ep["agent"], GOALS["unknown"])
    si = rec["t"]["step_idx"]
    chain = " -> ".join(ep["labels"][:si + 1][-8:])
    return (f"Stage goal: {goal}. Agent session (SOC security agent). Tool calls so far (last 8): {chain}. "
            f"Last action: {rec['cur']}.")

def mimo_call(rec):
    opts = ["<stop> (conclude session: no further tool call)" if o == STOP else o for o in rec["options"]]
    prompt = (state_text(rec) + "\nOptions:\n- " + "\n- ".join(opts) +
              "\nChoose exactly one option, replying with its bare action name "
              "(<stop> if concluding). One line only.")
    body = json.dumps({"model": "mimo26", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 2048, "temperature": 0.0}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(BASE, data=body, method="POST", headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.load(resp)
            dt = time.perf_counter() - t0
            txt = (out["choices"][0]["message"]["content"] or "").strip()
            pred = None
            for o in rec["options"]:
                if txt == o:
                    pred = o
                    break
            if pred is None:
                # tolerant match on mention of an option name
                for o in sorted(rec["options"], key=len, reverse=True):
                    if o != STOP and o in txt:
                        pred = o
                        break
                if pred is None and ("stop" in txt.lower() or "conclude" in txt.lower()):
                    pred = STOP
            return {"pred": pred, "raw": txt[:120], "lat": round(dt, 3)}
        except Exception as e:
            if attempt == 4:
                return {"pred": None, "raw": f"ERR {type(e).__name__}", "lat": None}
            time.sleep(3 * (attempt + 1))

print(f"mimo26 replay over {len(te)} test decisions...")
MIMO_CACHE = OUT / "e3_mimo_stream.json"
mimo = {}
if MIMO_CACHE.exists():
    mimo = {int(k): v for k, v in json.loads(MIMO_CACHE.read_text(encoding="utf-8")).items()}
    print(f"mimo stream cached: {len(mimo)} decisions")
else:
    done = 0
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(mimo_call, rec): i for i, rec in enumerate(te)}
        for f in as_completed(futs):
            i = futs[f]
            mimo[i] = f.result()
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(te)} elapsed {time.perf_counter() - t_all:.0f}s", flush=True)
                MIMO_CACHE.write_text(json.dumps({str(k): v for k, v in mimo.items()}), encoding="utf-8")
    MIMO_CACHE.write_text(json.dumps({str(k): v for k, v in mimo.items()}), encoding="utf-8")

if not (OUT / "feats_laya_cls.npz").exists():
    print("feats_laya_cls.npz not found yet - mimo stream saved; re-run after probe extraction")
    raise SystemExit(0)

# ---- typed stream: retrain the frozen-encoder probe (action, ce+brier, seed 0) inline ----
meta = json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8"))
action_vocab = meta["action_vocab"]
A2I = {**action_vocab, STOP: len(action_vocab)}
tr = [json.loads(l) for l in (OUT / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
_sub = OUT / "train_sub_idx.json"
if _sub.exists():  # dormant: train_sub_idx.json is retired (file absent); the reported runs train on the full split
    tr = [tr[i] for i in json.loads(_sub.read_text(encoding="utf-8"))]
va = [json.loads(l) for l in (OUT / "e2_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
Fe = np.load(OUT / "feats_laya_cls.npz")
import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)

torch.manual_seed(0)
ytr = np.array([A2I[r["target"]] for r in tr])
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
    gl = logits[i][idx]
    gl = gl - gl.max()
    p = np.exp(gl); p /= p.sum()
    typed.append({"pred": opts[int(p.argmax())], "conf": float(p.max()), "probs": dict(zip(opts, p.tolist()))})

# ---- metrics ---------------------------------------------------------------
def evaluate(pred_fn, name):
    """pred_fn(i) -> predicted option or None (na decision)."""
    by_trace = defaultdict(list)
    n = len(te)
    match = 0
    mismatches_at = []
    for i, rec in enumerate(te):
        pred = pred_fn(i)
        ok = (pred == rec["target"])
        match += ok
        by_trace[rec["trace"]].append((i, ok))
    # strict survival: an episode survives only with zero divergences anywhere
    # gap: teacher-forced full accounting - every wrong hop is charged (echoed/hit
    # decisions accrue zero), under per-wrong-hop cost conventions c1=1, c3=3
    surv = 0
    gap = {"c1": 0.0, "c3": 0.0}
    n_eps = len(by_trace)
    for tid, decs in by_trace.items():
        dead = False
        for i, ok in decs:
            if not ok:
                dead = True
                gap["c1"] += 1.0   # wrong-hops cost 1
                gap["c3"] += 3.0   # cost_wrong_act convention (Laya cfg)
        surv += (not dead)
    return {
        "name": name,
        "decision_match": round(match / n, 4),
        "episodes": n_eps,
        "episode_survival": round(surv / n_eps, 4),
        "gap_per_episode": {"wrong_cost_1": round(gap["c1"] / n_eps, 4),
                            "wrong_cost_3": round(gap["c3"] / n_eps, 4)},
    }

lats = sorted(v["lat"] for v in mimo.values() if v["lat"])
res = {
    "mimo26_everyhop": evaluate(lambda i: mimo[i]["pred"], "mimo26_everyhop"),
    "typed_only": evaluate(lambda i: typed[i]["pred"], "typed_only"),
}
res["mimo26_everyhop"]["lat_p50_s"] = lats[len(lats) // 2] if lats else None
res["mimo26_everyhop"]["lat_p90_s"] = lats[int(len(lats) * 0.9)] if lats else None
res["mimo26_everyhop"]["llm_calls_per_decision"] = 1.0

for tau in (0.55, 0.75, 0.90):
    esc = sum(1 for t in typed if t["conf"] < tau)
    def tau_fn(i, tau=tau):
        return mimo[i]["pred"] if typed[i]["conf"] < tau else typed[i]["pred"]
    r = evaluate(tau_fn, f"typed+esc@tau{tau}")
    r["escalation_rate"] = round(esc / len(te), 4)
    r["llm_calls_per_decision"] = round(esc / len(te), 4)
    res[f"typed+esc@tau{tau}"] = r

null_preds = sum(1 for v in mimo.values() if v["pred"] is None)
res["mimo_unparsed"] = null_preds
(OUT / "e3_replay.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))