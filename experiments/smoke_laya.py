# Smoke test: Laya as the typed-decision (Jev-style) next-hop router.
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # CN fallback if hub is unreachable

from laya import Router

state = (
    "Session soc-prod-event-task-9324. Goal: detection-stage triage of SIEM alert 78173 "
    "(possible brute force on host web-prod-03). Actions taken: siem_threat_read(78173), "
    "siem_alert_read(79277). Choose the next hop."
)
questions = {
    "next_hop": {
        "type": "choice",
        "instructions": "Which edge/action should the SOC detection agent take next?",
        "criteria": {
            "e07_siem_search": "search raw SIEM logs for the alert window",
            "e12_ti_enrich": "enrich the indicator with threat intelligence",
            "e33_update_goal": "conclude and write the triage report",
        },
    }
}

t0 = time.perf_counter()
router = Router()
t_load = time.perf_counter() - t0

t0 = time.perf_counter()
result = router.predict(state, questions)
dt = time.perf_counter() - t0

print("=== routing info ===")
print(json.dumps(result.get("routing", {}), ensure_ascii=False, indent=2))
print("=== answers ===")
print(json.dumps(result.get("answers", {}), ensure_ascii=False, indent=2)[:2000])
print(f"\nload_s={t_load:.2f} first_predict_s={dt:.3f}")

# warm latency over 30 calls
ts = []
for i in range(30):
    t0 = time.perf_counter()
    router.predict(state, questions)
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"predict latency (s, n=30): p50={ts[15]:.4f} p90={ts[27]:.4f} min={ts[0]:.4f} max={ts[-1]:.4f}")

# what answer payload keys exist (distribution/calibration surface)
ans = result["answers"]["next_hop"]
print("answer keys:", list(ans.keys()))
print("meta keys:", list(result.keys()))
print("meta:", json.dumps({k: v for k, v in result.items() if k not in ("answers",)}, ensure_ascii=False)[:600])