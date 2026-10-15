# Real-corpus pipeline: raw Langfuse observations -> folded episodes (paper §3 fold rules).
# Output: real/episodes.jsonl + real/corpus_census.json (frozen results; never regenerate blindly).
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path(Path(__file__).resolve().parents[1], "数据", "agent轨迹原始记录", "observations_raw.json")
OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
OUT.mkdir(parents=True, exist_ok=True)

data = json.loads(RAW.read_text(encoding="utf-8"))

# ---- normalize tool calls into per-trace step sequences -------------------
def norm_label(name: str) -> str:
    n = name
    if n.startswith("tool."):
        n = n[5:]
    return n

traces = defaultdict(list)
for e in data:
    if e["type"] != "TOOL":
        continue
    m = e.get("metadata") or {}
    rec = {
        "id": e["id"],
        "trace": e["traceId"],
        "session": m.get("session_id") or e["traceId"],
        "agent": m.get("agent_id") or "unknown",
        "t0": e["startTime"],
        "t1": e["endTime"],
        "latency": e.get("latency"),
        "label": norm_label(e["name"]),
        "input": e.get("input"),
    }
    traces[e["traceId"]].append(rec)

episodes = []
for tid, steps in traces.items():
    steps.sort(key=lambda r: (r["t0"], r["id"]))
    episodes.append({"trace": tid, "steps": steps})

episodes.sort(key=lambda x: x["steps"][0]["t0"])

# ---- fold loop events (§3): immediate retry -> retry; visited revisit -> back
# Implementation lives in fold_rules.py (side-effect free, shared with the
# public-corpus replication); semantics NOTE travels with it.
from fold_rules import fold  # noqa: E402

rows = []
for ep in episodes:
    seq = [s["label"] for s in ep["steps"]]
    kept, events = fold(seq)
    adv = [ev for ev in events if ev["kind"] == "advance"]
    back = [ev for ev in events if ev["kind"] == "back"]
    retry = [ev for ev in events if ev["kind"] == "retry"]
    backedges = [(ev["prev"], ev["label"]) for ev in back if ev["prev"] is not None and ev["prev"] != ev["label"]]
    agents = Counter(s["agent"] for s in ep["steps"])
    rows.append({
        "trace": ep["trace"],
        "session": ep["steps"][0]["session"],
        "agent": agents.most_common(1)[0][0],
        "t0": ep["steps"][0]["t0"],
        "n_steps": len(seq),
        "labels": seq,
        "kept": kept,
        "events": [ev["kind"] for ev in events],
        "n_advance": len(adv),
        "n_back": len(back),
        "n_retry": len(retry),
        "backedges": backedges,
        "latencies": [s["latency"] for s in ep["steps"]],
        "inputs": [s["input"] for s in ep["steps"]],
    })

# ---- census --------------------------------------------------------------
cat = Counter(lab for r in rows for lab in set(r["labels"]))
step_cat = Counter(lab for r in rows for lab in r["labels"])
agents = Counter(r["agent"] for r in rows)

fwd_edges = Counter()
for r in rows:
    kept = r["kept"]
    for a, b in zip(kept, kept[1:]):
        if a != b:
            fwd_edges[(a, b)] += 1
back_edge_set = Counter()
for r in rows:
    back_edge_set.update(tuple(e) for e in r["backedges"])

laps = sorted(Counter(r["n_steps"] for r in rows).items())
sizes = [r["n_steps"] for r in rows]
sizes_sorted = sorted(sizes)

census = {
    "source": "observations_raw.json",
    "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
    "n_obs": len(data),
    "n_tool_calls": sum(r["n_steps"] for r in rows),
    "n_episodes": len(rows),
    "n_sessions": len(set(r["session"] for r in rows)),
    "agents": dict(agents),
    "catalog_size": len(step_cat),
    "catalog_by_steps": step_cat.most_common(),
    "catalog_by_episodes": cat.most_common(),
    "steps_per_episode": {
        "min": sizes_sorted[0], "p50": sizes_sorted[len(sizes_sorted) // 2],
        "p90": sizes_sorted[int(len(sizes_sorted) * 0.9)], "max": sizes_sorted[-1],
        "mean": sum(sizes) / len(sizes),
    },
    "loop_events": {
        "retry": sum(r["n_retry"] for r in rows),
        "back": sum(r["n_back"] for r in rows),
    },
    "advance_hops": sum(r["n_advance"] for r in rows),
    "fwd_edges": len(fwd_edges),
    "back_edges": len(back_edge_set),
    "backedge_weight_classes_needed": True,
}

(OUT / "episodes.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
(OUT / "corpus_census.json").write_text(json.dumps(census, ensure_ascii=False, indent=2), encoding="utf-8")

print("episodes:", len(rows), "| calls:", census["n_tool_calls"], "| catalog:", census["catalog_size"])
print("agents:", dict(agents))
print("steps/ep:", census["steps_per_episode"])
print("loops:", census["loop_events"], "| advance:", census["advance_hops"])
print("fwd_edges:", len(fwd_edges), "| back_edges:", len(back_edge_set))
print("top labels:", step_cat.most_common(15))