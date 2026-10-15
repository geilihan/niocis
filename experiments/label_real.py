# Derived outcome labels for real episodes (paper-disclosed derivation):
#   proxy_a  = structural: episode ends in the report action update_goal (or within last 2 steps)
#   judge_b  = h200 verdict over action sequence + args vs stage goal (temperature 0)
#   human    = hand-audited subset (added post-hoc in outcome_labels.jsonl)
# Writes real/outcome_labels.jsonl (frozen) + calibration summary.
import os  # R-1
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
BASE = os.environ.get("ARIADNE_LLM_BASE",
       "https://<your-gateway-host>/v1/chat/completions")  # R-1: endpoint from env (SECURITY-NOTES.md)
KEY = os.environ.get("ARIADNE_LLM_KEY", "")  # R-1: rotate the leaked key; never commit one

GOALS = {
    "detection": ("SOC pipeline stage DETECTION. Goal: query logs to gather evidence and produce a "
                  "security triage disposition (false positive / manageable risk / needs action / uncertain), "
                  "written as the final event report via update_goal."),
    "protection": ("SOC pipeline stage PROTECTION. Goal: based on the triage result, execute or prepare "
                   "protective handling (blocking, isolation, rule updates) and record the handling summary "
                   "via update_goal."),
    "response": ("SOC pipeline stage RESPONSE. Goal: investigate and respond to the incident: gather "
                 "attack-path evidence, execute response actions, and record the response report via update_goal."),
    "recognition": ("SOC pipeline stage RECOGNITION. Goal: identify and profile the entities in the alert "
                    "(IPs, hosts, accounts), summarize the recognition report via update_goal."),
    "dlp-dispose": ("DLP disposal. Goal: handle the DLP alert to completion and record the disposal "
                    "summary via update_goal."),
    "QwenPaw_QA_Agent_0.2": ("QA agent. Goal: answer the user's question and conclude."),
    "unknown": ("Generic agent. Goal: complete the current task and conclude via update_goal."),
}

rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def short_input(obj):
    try:
        s = json.dumps(obj, ensure_ascii=False) if obj is not None else ""
    except Exception:
        s = str(obj)
    return s[:160]


def episode_text(r):
    lines = [f"{i + 1}. {lab}({short_input(inp)})" for i, (lab, inp) in enumerate(zip(r["labels"], r["inputs"]))]
    return "\n".join(lines)


def proxy_a(r):
    tail = r["labels"][-2:]
    return "update_goal" in tail


def judge_one(r):
    goal = GOALS.get(r["agent"], GOALS["unknown"])
    prompt = (
        f"Stage goal: {goal}\n\n"
        f"Session action log (tool calls in order, truncated args):\n{episode_text(r)}\n\n"
        f"Question: considering the structure of the session and its arguments, did the agent carry its "
        f"stage goal to completion (gathered/processed what the goal requires and closed with a concluding "
        f"report action)?\n"
        f'Reply as JSON only: {{"completed": true|false, "evidence": "brief reason referencing concrete steps"}}'
    )
    body = json.dumps({
        "model": "h200",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1600, "temperature": 0.0,
    }).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(BASE, data=body, method="POST", headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.load(resp)
            dt = time.perf_counter() - t0
            txt = out["choices"][0]["message"]["content"]
            start, end = txt.find("{"), txt.rfind("}")
            verdict = json.loads(txt[start:end + 1])
            return {"completed": bool(verdict.get("completed")),
                    "evidence": str(verdict.get("evidence", ""))[:300],
                    "judge_latency_s": round(dt, 3)}
        except Exception as e:
            if attempt == 3:
                return {"completed": None, "evidence": f"judge_error: {type(e).__name__}",
                        "judge_latency_s": None}
            time.sleep(2 + attempt * 2)


results = {}
done = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(judge_one, r): r["trace"] for r in rows}
    for f in as_completed(futs):
        tid = futs[f]
        results[tid] = f.result()
        done += 1
        if done % 200 == 0:
            print(f"judged {done}/{len(rows)}", flush=True)

out_rows = []
for r in rows:
    j = results[r["trace"]]
    out_rows.append({
        "trace": r["trace"], "session": r["session"], "agent": r["agent"],
        "n_steps": r["n_steps"],
        "proxy_a_completed": proxy_a(r),
        "judge_b_completed": j["completed"],
        "judge_evidence": j["evidence"],
        "judge_latency_s": j["judge_latency_s"],
    })

(OUT / "outcome_labels.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows), encoding="utf-8")

n = len(out_rows)
agree = sum(1 for r in out_rows if (r["proxy_a_completed"] == r["judge_b_completed"]))
errs = sum(1 for r in out_rows if r["judge_b_completed"] is None)
pa = sum(1 for r in out_rows if r["proxy_a_completed"])
jb = sum(1 for r in out_rows if r["judge_b_completed"] is True)
lats = sorted(r["judge_latency_s"] for r in out_rows if r["judge_latency_s"])
summary = {
    "n": n, "proxy_a_positive": pa, "judge_b_positive": jb, "agreement": agree,
    "agreement_rate": round(agree / n, 4), "judge_errors": errs,
    "judge_latency_p50_s": lats[len(lats) // 2] if lats else None,
}
(OUT / "outcome_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

# dump a 30-row audit sample keyed for manual inspection
sample = out_rows[:: max(1, n // 30)][:30]
(OUT / "audit_sample.json").write_text(
    json.dumps([{**r, "log": episode_text(next(x for x in rows if x["trace"] == r["trace"]))}
                for r in sample], ensure_ascii=False, indent=2), encoding="utf-8")
print("audit sample written:", len(sample))