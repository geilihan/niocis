# E2 Laya arm: zero-shot typed-choice over each decision's declared option space.
# Calibration: shipped checkpoint temperatures invalid (laya warns) -> fit scalar T on VAL
# (q = softmax(log p / T), NLL), then ECE/risk-coverage on TEST with calibrated probs.
# Tool descriptions are recovered from raw GENERATION tool declarations where available.
import json
import math
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from laya.agent import Agent

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
RAW = Path(Path(__file__).resolve().parents[1], "数据", "agent轨迹原始记录", "observations_raw.json")
LOCAL = str(Path(Path(__file__).resolve().parents[1], "experiments", "real", "models", "laya"))

# ---- tool descriptions from GENERATION tool declarations -------------------
desc = {}
raw = json.loads(RAW.read_text(encoding="utf-8"))
for e in raw:
    if e["type"] != "GENERATION":
        continue
    inp = e.get("input")
    tools = []
    if isinstance(inp, dict):
        tools = inp.get("tools") or []
    elif isinstance(inp, list):
        for msg in inp:
            c = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        tools.append({"function": blk})
    for t in tools:
        fn = t.get("function") or {}
        nm = fn.get("name")
        if nm and nm not in desc:
            d = fn.get("description") or ""
            d = re.sub(r"\s+", " ", d)[:220]
            desc[nm] = d
print("tool descriptions recovered:", len(desc))

GOALS = {
    "detection": "detect-stage: gather log evidence for the alert and judge triage disposition",
    "protection": "protect-stage: execute protective handling and record summary",
    "response": "response-stage: investigate attack path and act, record response report",
    "recognition": "recognize-stage: profile entities and record recognition report",
    "dlp-dispose": "dispose-stage: handle DLP alert and record disposal summary",
    "QwenPaw_QA_Agent_0.2": "answer the user question",
    "unknown": "complete the current task",
}

def load_split(name):
    return [json.loads(l) for l in (OUT / f"e2_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

va, te = load_split("val"), load_split("test")
rows = {r["trace"]: r for r in
        (json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
labels_meta = {}
for r in rows.values():
    labels_meta.setdefault(r["trace"], r)

trace_agent = {t: r["agent"] for t, r in rows.items()}

def state_text(rec):
    ep = rows[rec["trace"]]
    goal = ep["agent"]
    si = rec["t"]["step_idx"]
    chain = ">".join(x.split("__")[-1] for x in ep["labels"][:si + 1][-8:])
    return f"<{goal}> {chain} | next?"   # terse form shared with the probe arm

def question_of(rec):
    crit = {}
    for o in rec["options"]:
        if o == "<stop>":
            crit["<stop>"] = "conclude"
        else:
            crit[o] = o.split("__")[-1].replace("_", " ")   # names-only: cheap head
    return {"next": {"type": "choice",
                     "instructions": "What should the agent do next? Choose exactly one action.",
                     "criteria": crit}}

print("loading local laya...")
agent = Agent(LOCAL, device="cpu")

CACHE = OUT / "e2_laya_zeroshot_cache.json"

def run_split(recs, tag):
    """Returns list of decision records with per-call latencies; checkpoints every 50."""
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    out = []
    t_all = time.perf_counter()
    for i, rec in enumerate(recs):
        key = f"{tag}:{i}"
        if key in cache:
            out.append(cache[key])
            continue
        st = state_text(rec)
        q = question_of(rec)
        t0 = time.perf_counter()
        try:
            r = agent.predict(st, q)
            dt = time.perf_counter() - t0
            ans = r["answers"]["next"]
            row = {"trace": rec["trace"], "target": rec["target"],
                   "probs": ans.get("probabilities", {}), "choice": ans.get("choice"),
                   "in_opts": rec["target_in_opts"], "lat": round(dt, 4),
                   "cur": rec["cur"], "n_opts": len(rec["options"])}
        except Exception as ex:
            row = {"trace": rec["trace"], "target": rec["target"], "probs": {},
                   "choice": None, "in_opts": rec["target_in_opts"], "lat": None,
                   "err": type(ex).__name__, "cur": rec["cur"], "n_opts": len(rec["options"])}
        out.append(row)
        cache[key] = row
        if (i + 1) % 50 == 0:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[{tag}] {i + 1}/{len(recs)} {time.perf_counter() - t_all:.0f}s", flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out

val_out = run_split(va, "val")
te_out = run_split(te, "test")
(OUT / "e2_laya_raw.json").write_text(json.dumps({"val": val_out, "test": te_out}, ensure_ascii=False), encoding="utf-8")

# ---- temperature fit on VAL (q = softmax(log p / T)) ----------------------
def nll_at(T, recs):
    tot, n = 0.0, 0
    for r in recs:
        p = r["probs"]
        if not p or r["target"] not in p:
            continue
        keys = list(p.keys())
        lp = np.array([math.log(max(p[k], 1e-12)) / T for k in keys])
        lp -= lp.max()
        q = np.exp(lp); q /= q.sum()
        tot += -math.log(max(q[keys.index(r["target"])], 1e-12))
        n += 1
    return tot / max(1, n)

Ts = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
best_T, best_nll = 1.0, 1e9
for T in Ts:
    v = nll_at(T, val_out)
    print(f"T={T}: val NLL={v:.4f}")
    if v < best_nll:
        best_T, best_nll = T, v
print("chosen T:", best_T)

def metrics(recs, T):
    n = 0; ok = 0; confs = []; cors = []
    for r in recs:
        p = r["probs"]
        if not p or r["target"] not in p:
            continue
        keys = list(p.keys())
        lp = np.array([math.log(max(p[k], 1e-12)) / T for k in keys])
        lp -= lp.max()
        q = np.exp(lp); q /= q.sum()
        j = int(q.argmax())
        pred_ok = (keys[j] == r["target"])
        ok += pred_ok
        n += 1
        confs.append(float(q[j])); cors.append(pred_ok)
    rows_ = sorted(zip(confs, cors), key=lambda x: -x[0])
    ece = 0.0
    ca = np.array(confs); co = np.array(cors, dtype=float)
    for lo in range(15):
        hi = (lo + 1) / 15
        m = (ca > lo / 15) & (ca <= hi) if lo else (ca >= 0) & (ca <= hi)
        if m.sum():
            ece += m.mean() * abs(ca[m].mean() - co[m].mean())
    def risk_at(cov):
        k = max(1, int(round(cov * n)))
        return round(sum(1 for c, o in rows_[:k] if not o) / k, 5)
    lats = sorted(r["lat"] for r in recs if r["lat"])
    return {
        "n_eval": n,
        "acc": round(ok / n, 4),
        "ece15": round(float(ece), 4),
        "risk_at_cov": {str(c): risk_at(c) for c in (0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0)},
        "lat_p50_s": lats[len(lats) // 2] if lats else None,
        "lat_p90_s": lats[int(len(lats) * 0.9)] if lats else None,
    }

summary = {
    "temperature": best_T,
    "val": metrics(val_out, best_T),
    "test": metrics(te_out, best_T),
    "n_tool_desc": len(desc),
}
(OUT / "e2_laya_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))