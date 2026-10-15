# R42 fact checks (read-only): forward-pair bidirectionality, slice residual, option supply.
import json
import sys

sys.path.insert(0, str(Path(Path(__file__).resolve().parents[1], "experiments")))
from fold_rules import fold  # side-effect free by design

rows = [json.loads(l) for l in open(str(Path(Path(__file__).resolve().parents[1], "experiments", "real", "episodes.jsonl")), encoding="utf-8")]
print("episodes:", len(rows))
print("keys:", sorted(rows[0].keys()))

fwd, ret = set(), set()
slices = {}
for r in rows:
    labs = r.get("labels") or r.get("label_seq") or r.get("fold") or r.get("actions")
    if labs is None:
        print("row keys:", sorted(r.keys()))
        break
    if isinstance(labs[0], dict):
        labs = [x["label"] for x in labs]
    kept, ev = fold(labs)
    for e in ev:
        if e["kind"] == "advance" and e["prev"] is not None:
            fwd.add((e["prev"], e["label"]))
        elif e["kind"] == "back":
            ret.add((e["prev"], e["label"]))
    ag = r.get("agent") or r.get("slice") or r.get("domain") or "n/a"
    slices[ag] = slices.get(ag, 0) + 1

union = fwd | ret
both = fwd & ret
print("fwd", len(fwd), "ret", len(ret), "union", len(union), "both", len(both), "pure_ret", len(ret - fwd))
bi = [(a, b) for (a, b) in fwd if a != b and (b, a) in fwd]
print("bidirectional forward pairs:", len(bi), "-> arcs in 2-cycles:", 2 * len(bi))
selfloops = [(a, b) for (a, b) in fwd if a == b]
print("forward self-pairs:", len(selfloops))
print("slices:", slices)