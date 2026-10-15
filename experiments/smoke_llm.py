# Smoke test + latency measurement for gateway models mimo26 (LLM opponent) and h200 (judge).
import os  # R-1
import json
import time
import urllib.request

BASE = os.environ.get("ARIADNE_LLM_BASE",
       "https://<your-gateway-host>/v1/chat/completions")  # R-1: endpoint from env (SECURITY-NOTES.md)
KEY = os.environ.get("ARIADNE_LLM_KEY", "")  # R-1: rotate the leaked key; never commit one

def call(model: str, messages, max_tokens=64, temperature=0.0):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(BASE, data=body, method="POST", headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0

# mimo26: next-hop style opponent probe
msgs_opp = [
    {"role": "system", "content": "You are a SOC security-triage agent choosing the next tool action. Reply with exactly one action name."},
    {"role": "user", "content": "Session: SIEM alert 78173 possible brute force on web-prod-03. Done: siem_threat_read(78173). "
                                "Options: siem_alert_read | siem_search | ti_enrich | update_goal. Which is next? One action name only."},
]
# h200: judge style probe with JSON verdict
msgs_judge = [
    {"role": "system", "content": "You are a strict evaluator. Reply with JSON only."},
    {"role": "user", "content": 'Task: "triage SIEM alert 78173 and conclude false-positive or actionable". '
                                'Trace: siem_threat_read -> siem_search -> update_goal(report=actionable, brute force confirmed). '
                                'Did the task succeed? Reply {"success": true|false, "reason": "..."}'},
]

for model, msgs in (("mimo26", msgs_opp), ("h200", msgs_judge)):
    for probe in range(1, 11):
        try:
            out, dt = call(model, msgs)
            ch = out["choices"][0]
            txt = (ch.get("message", {}).get("content") or "")[:160].replace("\n", " ")
            print(f"{model} probe{probe:02d}: {dt:.3f}s | finish={ch.get('finish_reason')} | usage={out.get('usage', {}).get('total_tokens')} | {txt}")
        except Exception as e:
            print(f"{model} probe{probe:02d}: ERROR {type(e).__name__}: {str(e)[:160]}")