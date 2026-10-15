# Smoke: load Laya from the LOCAL ModelScope-downloaded dir, run typed choice on a real-style state.
import json
import time
from laya.agent import Agent

LOCAL = str(Path(Path(__file__).resolve().parents[1], "experiments", "real", "models", "laya"))

t0 = time.perf_counter()
agent = Agent(LOCAL, device="cpu")
print(f"load: {time.perf_counter() - t0:.2f}s")

state = (
    "SOC detection session soc-prod-event-task-9324. Stage goal: evidence triage of SIEM alert 78173 "
    "(possible brute force on host web-prod-03). Done so far: siem_threat_read({id:78173}), "
    "siem_alert_read({id:79277}). Pick the next tool call."
)
questions = {
    "next": {
        "type": "choice",
        "instructions": "Which action should the SOC detection agent take next?",
        "criteria": {
            "sec-siem__search": "search raw SIEM logs over the alert window",
            "sec-api-service__sec_api_service_prod-ti_enhance": "enrich indicator with threat intelligence",
            "update_goal": "conclude and write the triage report",
        },
    }
}

t0 = time.perf_counter()
res = agent.predict(state, questions)
dt = time.perf_counter() - t0
print(json.dumps(res, ensure_ascii=False, indent=2)[:3000])
print(f"first predict: {dt:.3f}s")

ts = []
for _ in range(30):
    t0 = time.perf_counter()
    agent.predict(state, questions)
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"warm predict (s, n=30): p50={ts[15]:.4f} p90={ts[27]:.4f} max={ts[-1]:.4f}")