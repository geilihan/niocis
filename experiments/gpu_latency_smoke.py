# Measure typed-decision latency on the actual deployment-class device (RTX A3000):
# (a) full Laya typed-choice predict over a representative ~17-option decision;
# (b) bare encoder forward (the hidden-state probe's per-decision marginal cost at batch 1).
# Writes real/gpu_latency.json (frozen).
import json
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
from laya.agent import Agent

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
LOCAL = str(Path(Path(__file__).resolve().parents[1], "experiments", "real", "models", "laya"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

state = ("<detection> alert_read>threat_read>tdp_timeline>ti_search>ti_geo>waf_attack_types>ban_ips>search | next?")
crit = {}
opts = ["sec-siem__search", "event_list_read", "siem_alert_read", "siem_threat_read", "tdp_timeline",
        "ti_search", "ti_geo", "ti_enhance", "waf_attack_types", "waf_attacked_domains", "get_ban_ips",
        "hids_query", "mde_kql", "ieg_owner", "asset_owner", "chat_with_agent", "<stop>"]
for o in opts:
    crit[o] = "conclude" if o == "<stop>" else o.split("__")[-1].replace("_", " ")
q = {"next": {"type": "choice", "instructions": "Next action?", "criteria": crit}}

agent = Agent(LOCAL, device=DEV)

# warm-up
for _ in range(3):
    agent.predict(state, q)
if DEV == "cuda":
    torch.cuda.synchronize()

lat = []
for _ in range(50):
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    agent.predict(state, q)
    if DEV == "cuda":
        torch.cuda.synchronize()
    lat.append(time.perf_counter() - t0)
lat.sort()
model, tok = agent.model, agent.tok
model.eval()

enc = []
with torch.inference_mode():
    batch = tok([state], padding=True, truncation=True, max_len=64, return_tensors="pt").to(DEV)
    for _ in range(3):
        model.encoder(**batch)
    if DEV == "cuda":
        torch.cuda.synchronize()
    for _ in range(50):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.encoder(**batch)
        if DEV == "cuda":
            torch.cuda.synchronize()
        enc.append(time.perf_counter() - t0)
enc.sort()

result = {
    "device": DEV,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "typed_choice_predict_ms": {"p50": round(1000 * lat[25], 2), "p90": round(1000 * lat[45], 2),
                                "min": round(1000 * lat[0], 2), "max": round(1000 * lat[-1], 2), "n": 50,
                                "n_options": len(opts)},
    "encoder_forward_ms": {"p50": round(1000 * enc[25], 2), "p90": round(1000 * enc[45], 2), "n": 50},
}
(OUT / "gpu_latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))