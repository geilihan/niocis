# Deep probe: what usable signal is in observations_raw.json, what is absent.
import json
from collections import Counter
from pathlib import Path

P = Path(Path(__file__).resolve().parents[1], "数据", "agent轨迹原始记录", "observations_raw.json")
data = json.loads(P.read_text(encoding="utf-8"))

print("n records:", len(data))
print("type:", Counter(e["type"] for e in data))
print("level:", Counter(e["level"] for e in data))
print("model:", Counter(e["model"] for e in data))
print("environment:", Counter(e["environment"] for e in data))

traces = Counter(e["traceId"] for e in data)
print("unique traceId:", len(traces))
sizes = sorted(traces.values())
print("obs per trace: min", sizes[0], "p50", sizes[len(sizes)//2], "p90", sizes[int(len(sizes)*0.9)], "max", sizes[-1])

sess = Counter()
agent = Counter()
user = Counter()
chan = Counter()
for e in data:
    m = e.get("metadata") or {}
    sess[m.get("session_id")] += 1
    agent[m.get("agent_id")] += 1
    user[m.get("user_id")] += 1
    chan[m.get("channel")] += 1
print("unique session_id:", len(sess))
print("agent_id:", dict(agent))
print("user_id:", dict(user))
print("channel:", dict(chan))

names = Counter(e["name"] for e in data)
print("\ntop name values:")
for k, v in names.most_common(30):
    print(f"  {v:6d}  {k}")

nulls = {k: sum(1 for e in data if e.get(k) in (None, "", {}, [])) for k in
         ("input", "output", "latency", "statusMessage", "promptId", "completionStartTime")}
print("\nnull counts:", nulls)

lat = [e["latency"] for e in data if isinstance(e.get("latency"), (int, float))]
if lat:
    lat_s = sorted(lat)
    print(f"latency (s): min {lat_s[0]:.3f} p50 {lat_s[len(lat_s)//2]:.3f} p90 {lat_s[int(len(lat_s)*0.9)]:.3f} max {lat_s[-1]:.3f}")

# metadata key census
mk = Counter()
for e in data:
    mk.update((e.get("metadata") or {}).keys())
print("\nmetadata key census:", dict(mk))

# samples of GENERATION/LLM-ish records and non-TOOL records
nong = [e for e in data if e["type"] != "TOOL"]
print("\nnon-TOOL count:", len(nong))
for e in nong[:2]:
    slim = {k: e[k] for k in ("id", "type", "name", "model", "input", "output", "latency", "statusMessage")}
    print(json.dumps(slim, ensure_ascii=False)[:2000])
    print("---")

# a TOOL record with non-null output if any
with_out = next((e for e in data if e["type"] == "TOOL" and e.get("output")), None)
print("\nTOOL with non-null output:", bool(with_out))
if with_out:
    print(json.dumps({k: with_out[k] for k in ("name", "input", "output", "latency")}, ensure_ascii=False)[:1500])

# statusMessage non-null examples (possible errors/outcomes)
sm = [e for e in data if e.get("statusMessage")]
print("\nnon-null statusMessage:", len(sm))
for e in sm[:5]:
    print("  ", e["type"], e["name"][:60], "|", str(e["statusMessage"])[:200])

# input shapes for TOOL vs others
print("\ninput type census:", Counter(type(e["input"]).__name__ for e in data))