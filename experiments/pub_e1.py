# E1 saturation replication on PUBLIC event logs. Uses the SAME fold()
# implementation as the SOC corpus (parse_real_corpus.py) so the fold rules of
# §3 are literally shared. Per corpus: conservation self-checks + saturation
# curves (20 without-replacement subsamples per size where n allows).
# Writes runs/pub_e1_results.json.
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_rules import fold  # noqa: E402  the one true fold (shared with the SOC pipeline)

# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container
_HOST = Path("/lw2/pubdata")
PUB = _HOST if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
RUNS = Path("/lw2/runs") if Path("/lw2/runs").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stats(cases):
    """Fold totals + per-episode pair roles for one collection of label seqs."""
    n_adv = n_retry = n_back = 0
    fwd, backp = set(), set()
    labels = set()
    per_ep = []
    for seq in (c["labels"] for c in cases):
        kept, events = fold(seq)
        adv = [e for e in events if e["kind"] == "advance"]
        ret = [e for e in events if e["kind"] == "back"]
        rep = [e for e in events if e["kind"] == "retry"]
        assert len(adv) + len(ret) + len(rep) == len(seq), "fold conservation broke"
        n_adv += len(adv); n_retry += len(rep); n_back += len(ret)
        labels.update(seq)
        pairs = [(e["prev"], e["label"]) for e in adv if e["prev"] is not None]
        rpairs = [(e["prev"], e["label"]) for e in ret if e["prev"] is not None and e["prev"] != e["label"]]
        fwd.update(pairs); backp.update(rpairs)
        per_ep.append((set(kept), set(pairs), set(rpairs), len(rep), len(ret), len(seq)))
    steps = sum(c["count"] for c in cases)
    return {
        "cases": len(cases), "events": steps, "labels": len(labels),
        "advance": n_adv, "retry": n_retry, "back": n_back,
        "loop_mass": round((n_retry + n_back) / steps, 4),
        "fwd_pairs": len(fwd), "back_pairs": len(backp),
        "steps_per_case_mean": round(steps / max(1, len(cases)), 2),
    }, per_ep


def saturation(cases, per_ep, sizes, reps, seed0):
    out = {}
    n = len(cases)
    for k in sizes:
        agg = {"nodes": [], "fwd": [], "back": [], "loops_per_100_steps": [], "back_per_100_eps": []}
        for r in range(reps):
            idx = np.random.RandomState(seed0 * 1000 + r).choice(n, size=min(k, n), replace=False)
            nodes, fwd, backp = set(), set(), set()
            steps = retry = backn = 0
            m = 0
            for i in idx:
                kept, pr, rp, nr, nb, ln = per_ep[i]
                nodes.update(kept); fwd.update(pr); backp.update(rp)
                steps += ln; retry += nr; backn += nb; m += 1
            agg["nodes"].append(len(nodes))
            agg["fwd"].append(len(fwd))
            agg["back"].append(len(backp))
            agg["loops_per_100_steps"].append(100 * (retry + backn) / max(1, steps))
            agg["back_per_100_eps"].append(100 * backn / max(1, m))
        out[str(k)] = {key: {"p50": round(float(np.median(v)), 2),
                             "min": round(float(np.min(v)), 2),
                             "max": round(float(np.max(v)), 2)} for key, v in agg.items()}
    return out


results = {}
for p in sorted(PUB.glob("*_cases.jsonl")):
    name = p.name.replace("_cases.jsonl", "")
    cases = read_jsonl(p)
    tot, per_ep = stats(cases)
    n = len(cases)
    reps = 20 if n >= 50 else 5
    sizes = sorted({max(5, int(x)) for x in np.unique(
        np.round(np.geomspace(min(20, n), n, 12)).astype(int))})
    results[name] = {"totals": tot,
                     "curve": saturation(cases, per_ep, sizes, reps, seed0=1000),
                     "sizes": sizes, "reps": reps}
    print(f"{name:12s} totals {tot}  sizes {sizes[0]}..{sizes[-1]}")

(RUNS / "pub_e1_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("wrote", RUNS / "pub_e1_results.json")