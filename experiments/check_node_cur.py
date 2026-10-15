# Probe: is t.node injective w.r.t. cur label? (grounds Markov context choice)
# read-only diagnostic on decision rows; writes nothing.
import json
from collections import defaultdict
from pathlib import Path

_pub = Path("/lw2/pubdata") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
_runs = Path("/lw2/runs") if Path("/lw2/runs").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")

for tag, p in [("soc", _runs / "e2_train.jsonl")] + [
        (m.name.replace("_e2_meta.json", ""), _pub / (m.name.replace("_e2_meta.json", "") + "_e2_train.jsonl"))
        for m in sorted(_pub.glob("*_e2_meta.json"))]:
    if not p.exists():
        continue
    cur2nodes, node2curs = defaultdict(set), defaultdict(set)
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        cur2nodes[r["cur"]].add(r["t"]["node"])
        node2curs[r["t"]["node"]].add(r["cur"])
    ncur_multi = sum(1 for v in cur2nodes.values() if len(v) > 1)
    nnode_multi = sum(1 for v in node2curs.values() if len(v) > 1)
    print(f"{tag:14s} rows(cur)={len(cur2nodes):5d} nodes={len(node2curs):5d} "
          f"cur->multi-node={ncur_multi:4d} node->multi-cur={nnode_multi:4d}")