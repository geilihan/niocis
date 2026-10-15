# Inspect full response schema with a generous token budget.
import os  # R-1
import json
import time
import urllib.request

BASE = os.environ.get("ARIADNE_LLM_BASE",
       "https://<your-gateway-host>/v1/chat/completions")  # R-1: endpoint from env (SECURITY-NOTES.md)
KEY = os.environ.get("ARIADNE_LLM_KEY", "")  # R-1: rotate the leaked key; never commit one

def call(model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(BASE, data=body, method="POST", headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0

msgs = [
    {"role": "system", "content": "You are a SOC security-triage agent choosing the next tool action. Reply with exactly one action name."},
    {"role": "user", "content": "Session: SIEM alert 78173 possible brute force on web-prod-03. Done: siem_threat_read(78173). "
                                "Options: siem_alert_read | siem_search | ti_enrich | update_goal. Which is next? One action name only."},
]
for model in ("mimo26", "h200"):
    out, dt = call(model, msgs, 2048)
    print(f"=== {model} {dt:.3f}s ===")
    print(json.dumps(out, ensure_ascii=False)[:3000])
    print()