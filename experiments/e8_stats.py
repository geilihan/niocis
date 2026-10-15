# Forensics stats on the real folded corpus: retry-run lengths, back-edge revisit distances,
# per-agent/domain slices, loop-mass ratio. Pure counting; writes real/forensics_stats.json.
import json
from collections import Counter
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
labels = {}
lp = OUT / "outcome_labels.jsonl"
if lp.exists():
    labels = {json.loads(l)["trace"]: json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()}

retry_runs = Counter()   # consecutive same-label run length beyond 1
back_dist = Counter()    # steps since the target label was last kept, bucketed
loop_events = 0
advance = 0
per_agent = Counter()
per_agent_steps = Counter()

for r in rows:
    seq = r["labels"]
    kept_idx = {}
    run = 0
    for i, lab in enumerate(seq):
        # retry runs
        if i and lab == seq[i - 1]:
            run += 1
        else:
            if run:
                retry_runs[run] += 1
            run = 1
        # fold classification (§3)
        seen = seq[:i]
        if i and lab == seq[i - 1]:
            loop_events += 1
        elif lab in seen:
            loop_events += 1
            d = i - 1 - max(j for j, x in enumerate(seen) if x == lab)
            bucket = "1" if d <= 1 else ("2-3" if d <= 3 else ("4-8" if d <= 8 else "9+"))
            back_dist[bucket] += 1
        else:
            advance += 1
    if run:
        retry_runs[run] += 1
    per_agent[r["agent"]] += 1
    per_agent_steps[r["agent"]] += r["n_steps"]

n_unfolded = sum(r["n_steps"] for r in rows)
stats = {
    "episodes": len(rows),
    "unfolded_steps": n_unfolded,
    "advance_steps": advance,
    "loop_events": loop_events,
    "loop_mass_ratio": round(loop_events / n_unfolded, 4),
    "retry_run_len_dist": dict(sorted(retry_runs.items())),
    "back_revisit_distance_dist": dict(back_dist),
    "per_agent_episodes": dict(per_agent),
    "per_agent_steps": dict(per_agent_steps),
    "outcome_labels_loaded": bool(labels),
}
if labels:
    done = sum(1 for t in (r["trace"] for r in rows)
               if t in labels and labels[t].get("proxy_a_completed"))
    judged = sum(1 for t in (r["trace"] for r in rows)
                 if t in labels and labels[t].get("judge_b_completed") is True)
    stats["outcome"] = {"proxy_a_positive": done, "judge_b_positive": judged}

(OUT / "forensics_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))