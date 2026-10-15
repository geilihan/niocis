# Outcome labels v2: THREE-CLASS derived outcome (completed / blocked / abandoned) +
# h200 judge v2 whose criterion matches the close-out convention (calibration pass).
# Writes real/outcome_labels_v2.jsonl (frozen).
import os  # R-1
import json
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
BASE = os.environ.get("ARIADNE_LLM_BASE",
       "https://<your-gateway-host>/v1/chat/completions")  # R-1: endpoint from env (SECURITY-NOTES.md)
KEY = os.environ.get("ARIADNE_LLM_KEY", "")  # R-1: rotate the leaked key; never commit one

rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

def derived(r):
    labs, ins = r["labels"], r["inputs"]
    if labs and labs[-2:] and any(l == "update_goal" for l in labs[-2:]):
        tail = [(l, i) for l, i in zip(labs, ins)][-2:]
        for l, i in tail:
            if l == "update_goal":
                return "completed" if (i or {}).get("status") == "complete" else "blocked"
    return "abandoned"

def episode_text(r):
    out = []
    for i, (lab, inp) in enumerate(zip(r["labels"], r["inputs"])):
        try:
            s = json.dumps(inp, ensure_ascii=False) if inp is not None else ""
        except Exception:
            s = ""
        out.append(f"{i + 1}. {lab}({s[:140]})")
    return "\n".join(out)

GOALS = {
    "detection": "gather log evidence for the alert and reach a security triage disposition",
    "protection": "execute or prepare protective handling and close the handling loop",
    "response": "investigate the attack path, act, and record the response outcome",
    "recognition": "profile entities and record the recognition outcome",
    "dlp-dispose": "handle the DLP alert to closure",
    "QwenPaw_QA_Agent_0.2": "answer the question and conclude",
    "unknown": "carry the task to closure",
}

def judge_v2(r):
    goal = GOALS.get(r["agent"], GOALS["unknown"])
    prompt = (
        f"Stage goal: {goal}.\n\n"
        f"Session tool-call log (args truncated):\n{episode_text(r)}\n\n"
        "Operational convention for THIS log: a session is COMPLETED when it does substantive "
        "goal-directed work and ends with a declared close-out: `update_goal({\"status\": \"complete\"})` "
        "(optionally preceded by writing its conclusion via append_file or agent chat). The report "
        "body text is not stored inside update_goal; do not hold that against the session. "
        "A session is ABANDONED when it stops without any close-out act (truncated mid-work, or only "
        "pointless browsing), and BLOCKED when the close-out status is 'blocked'.\n"
        'Reply as JSON only: {"outcome": "completed"|"blocked"|"abandoned", "evidence": "brief reason"}'
    )
    body = json.dumps({"model": "h200",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1600, "temperature": 0.0}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(BASE, data=body, method="POST", headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.load(resp)
            dt = time.perf_counter() - t0
            txt = out["choices"][0]["message"]["content"]
            v = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            o = str(v.get("outcome", "")).lower()
            o = o if o in ("completed", "blocked", "abandoned") else None
            return {"judge_v2": o, "evidence": str(v.get("evidence", ""))[:220], "lat": round(dt, 3)}
        except Exception as e:
            if attempt == 3:
                return {"judge_v2": None, "evidence": f"judge_error {type(e).__name__}", "lat": None}
            time.sleep(2 * (attempt + 1))

print(f"judging {len(rows)} episodes with v2 criterion...", flush=True)
res = {}
done = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(judge_v2, r): r["trace"] for r in rows}
    for f in as_completed(futs):
        res[futs[f]] = f.result()
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(rows)}", flush=True)

out_rows = []
for r in rows:
    d = derived(r)
    j = res[r["trace"]]
    out_rows.append({
        "trace": r["trace"], "session": r["session"], "agent": r["agent"], "n_steps": r["n_steps"],
        "derived_outcome": d, "judge_v2": j["judge_v2"], "judge_evidence": j["evidence"],
        "judge_latency_s": j["lat"],
    })

(OUT / "outcome_labels_v2.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows), encoding="utf-8")

n = len(out_rows)
dist = Counter(r["derived_outcome"] for r in out_rows)
jd = Counter(r["judge_v2"] for r in out_rows if r["judge_v2"])
agree = sum(1 for r in out_rows if r["judge_v2"] == r["derived_outcome"])
by_agent = {}
for r in out_rows:
    by_agent.setdefault(r["agent"], Counter())[r["derived_outcome"]] += 1
lats = sorted(r["judge_latency_s"] for r in out_rows if r["judge_latency_s"])
summary = {
    "n": n,
    "derived_dist": dict(dist),
    "judge_v2_dist": dict(jd),
    "agreement": agree,
    "agreement_rate": round(agree / n, 4),
    "by_agent": {k: dict(v) for k, v in by_agent.items()},
    "judge_latency_p50_s": lats[len(lats) // 2] if lats else None,
    "judge_latency_p90_s": lats[int(len(lats) * 0.9)] if lats else None,
}
(OUT / "outcome_label_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))