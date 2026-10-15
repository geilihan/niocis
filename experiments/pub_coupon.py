# Occupancy (coupon) null for node-discovery saturation -- zero free parameters.
# Under case-level sampling WITHOUT replacement, label l with case-support c_l
# escapes an m-case subsample with probability exactly C(n-c_l, m)/C(n, m), so
#   E[D(m)] = sum_l [ 1 - prod_{i<m} (n - c_l - i)/(n - i) ].
# Published against the subsample medians of pub_e1_results.json (same 12
# sizes) and the SOC e1_saturation.json curve: if observed discovery follows
# the null, graph-size convergence is combinatorics, not domain luck, which is
# exactly what makes the compression claim portable.
import json
import math
from pathlib import Path

_HOST = Path("/lw2/pubdata")
PUB = _host = _HOST if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
RUNS = Path("/lw2/runs") if Path("/lw2/runs").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")


def p_unseen(n, c, m):
    # C(n-c, m) / C(n, m) via the product form; exact integer-free arithmetic
    if c <= 0:
        return 1.0
    if m > n - c:
        return 0.0
    p = 1.0
    for i in range(m):
        p *= (n - c - i) / (n - i)
        if p <= 0:
            return 0.0
    return p


def expected_distinct(n, support, m):
    return sum(1.0 - p_unseen(n, c, m) for c in support.values())


def load_cases(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def support_from(cases):
    out = {}
    for c in cases:
        for lab in set(c["labels"]):
            out[lab] = out.get(lab, 0) + 1
    return out


results = {}

# ---- public corpora: pair against pub_e1_results.json medians ---------------
e1 = json.loads((RUNS / "pub_e1_results.json").read_text(encoding="utf-8"))
for name in sorted(e1):
    if e1[name]["totals"]["cases"] < 20:
        print(f"{name:14s} (skipped: <20 cases, no curve)")
        continue
    cases = load_cases(PUB / f"{name}_cases.jsonl")
    n = len(cases)
    sup = support_from(cases)
    sizes = e1[name]["sizes"]
    pred = [expected_distinct(n, sup, m) for m in sizes]
    obs = [e1[name]["curve"][str(m)]["nodes"]["p50"] for m in sizes]
    diffs = [abs(a - b) for a, b in zip(pred, obs)]
    fit = {"n_cases": n, "K": len(sup), "sizes": sizes,
           "pred_nodes": [round(x, 2) for x in pred],
           "obs_nodes_p50": obs,
           "mad_nodes": round(sum(diffs) / len(diffs), 2),
           "max_abs_nodes": round(max(diffs), 2),
           "mad_over_K": round(sum(diffs) / len(diffs) / len(sup), 4)}
    results[name] = fit
    print(f"{name:14s} K={len(sup):3d} n={n:6d}  MAD {fit['mad_nodes']:5.2f} nodes "
          f"({100 * fit['mad_over_K']:4.1f}% of K)  max|dev| {fit['max_abs_nodes']:5.2f}")

# ---- SOC corpus: same null against e1_saturation.json medians ---------------
e1s_path = RUNS / "e1_saturation.json"
ep_path = RUNS / "episodes.jsonl"
if e1s_path.exists() and ep_path.exists():
    e1s = json.loads(e1s_path.read_text(encoding="utf-8"))
    # shape-tolerant read of the SOC curve: find the nodes-median accessor
    curve = e1s.get("curve") or {}
    def nodes_p50(entry):
        if isinstance(entry, dict):
            for key in ("nodes", "node", "semantic_nodes", "labels_seen", "distinct_labels"):
                v = entry.get(key)
                if isinstance(v, dict) and "p50" in v:
                    return float(v["p50"])
                if isinstance(v, (int, float)):
                    return float(v)
        return None
    sizes = sorted(int(k) for k in curve)
    obs = [nodes_p50(curve[str(m)]) for m in sizes]
    if any(o is None for o in obs):
        print("soc: curve shape unrecognized, emitting raw keys for inspection:",
              sorted(curve[str(sizes[0])].keys()) if sizes else "empty")
    else:
        eps = load_cases(ep_path)
        n = len(eps)
        sup = support_from(eps)
        pred = [expected_distinct(n, sup, m) for m in sizes]
        diffs = [abs(a - b) for a, b in zip(pred, obs)]
        results["soc"] = {"n_cases": n, "K": len(sup), "sizes": sizes,
                          "pred_nodes": [round(x, 2) for x in pred],
                          "obs_nodes_p50": obs,
                          "mad_nodes": round(sum(diffs) / len(diffs), 2),
                          "max_abs_nodes": round(max(diffs), 2),
                          "mad_over_K": round(sum(diffs) / len(diffs) / len(sup), 4)}
        print(f"{'soc':14s} K={len(sup):3d} n={n:6d}  MAD {results['soc']['mad_nodes']:5.2f} "
              f"({100 * results['soc']['mad_over_K']:4.1f}% of K)  "
              f"max|dev| {results['soc']['max_abs_nodes']:5.2f}")

(RUNS / "pub_coupon.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("wrote", RUNS / "pub_coupon.json")