# E1 on the real corpus: induction saturation (nodes, forward edges, back-edges) vs episodes.
# 20 without-replacement subsamples per size, seeds pinned; writes real/e1_saturation.json (frozen).
import json
import random
from collections import Counter
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

def fold_kept(labels):
    kept = []
    backs = []
    for lab in labels:
        if kept and lab == kept[-1]:
            continue                      # retry: loop event only
        if lab in kept:
            if kept[-1] != lab:
                backs.append((kept[-1], lab))
            kept.append(lab)
        else:
            kept.append(lab)
    return kept, backs

def graph_of(eps):
    nodes = set()
    fwd = set()
    back = set()
    n_steps = 0
    n_retry = 0
    n_backev = 0
    for r in eps:
        kept, backs = fold_kept(r["labels"])
        nodes.update(kept)
        for a, b in zip(kept, kept[1:]):
            if a != b:
                fwd.add((a, b))
        back.update(backs)
        n_steps += r["n_steps"]
        n_retry += r["n_retry"]
        n_backev += r["n_back"]
    loops_per_100_steps = 100.0 * (n_retry + n_backev) / max(1, n_steps)
    back_per_100_eps = 100.0 * n_backev / max(1, len(eps))
    return nodes, fwd, back, loops_per_100_steps, back_per_100_eps

sizes = [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1615]
sizes = sorted(set(s for s in sizes if s <= len(rows)))

curve = {}
for k in sizes:
    stat = {key: [] for key in ("nodes", "fwd", "back", "loops_per_100_steps", "back_per_100_eps")}
    for rep in range(20):
        rng = random.Random(1000 + rep)
        sample = rng.sample(rows, min(k, len(rows)))
        n, f, b, lps, bpe = graph_of(sample)
        stat["nodes"].append(len(n))
        stat["fwd"].append(len(f))
        stat["back"].append(len(b))
        stat["loops_per_100_steps"].append(round(lps, 2))
        stat["back_per_100_eps"].append(round(bpe, 2))
    curve[str(k)] = {
        key: {q: sorted(v)[idx] for q, idx in (("min", 0), ("p50", len(v) // 2), ("max", len(v) - 1))}
        for key, v in stat.items()
    }
    print(k, {key: curve[str(k)][key]["p50"] for key in ("nodes", "fwd", "back", "loops_per_100_steps", "back_per_100_eps")})

n, f, b, lps, bpe = graph_of(rows)
full = {"nodes": len(n), "fwd": len(f), "back": len(b),
        "loops_per_100_steps": round(lps, 2), "back_per_100_eps": round(bpe, 2),
        "node_names": sorted(n)}
(OUT / "e1_saturation.json").write_text(
    json.dumps({"curve": curve, "full": full}, ensure_ascii=False, indent=2), encoding="utf-8")
print("full graph:", full["nodes"], "nodes", full["fwd"], "fwd edges", full["back"], "back edges",
      "| loops/100steps", full["loops_per_100_steps"], "| back/100eps", full["back_per_100_eps"])