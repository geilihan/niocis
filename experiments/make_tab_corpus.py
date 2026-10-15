# Regenerate paper/tables/tab_corpus.tex entirely from frozen real/ artifacts.
import hashlib
import json
from collections import Counter
from pathlib import Path

_HOST = Path("/lw2/runs")
OUT = _HOST if _HOST.exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
TAB = ((_HOST / "tables") if _HOST.exists()
       else Path(Path(__file__).resolve().parents[1], "paper", "tables")) / "tab_corpus.tex"

rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
labels = {json.loads(l)["trace"]: json.loads(l) for l in
          (OUT / "outcome_labels_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

def split_of(trace):
    h = int(hashlib.sha256(trace.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 80 else "test")

agg = {}
for r in rows:
    sp = split_of(r["trace"])
    a = agg.setdefault(sp, Counter())
    a["episodes"] += 1
    a["completed"] += (labels[r["trace"]]["derived_outcome"] == "completed")
    a["steps"] += r["n_steps"]
    a["advance"] += r["n_advance"]
    a["loop"] += r["n_retry"] + r["n_back"]

order = ["train", "val", "test", "all"]
allc = Counter()
for a in agg.values():
    allc.update(a)

lines = [r"\begin{tabular}{lrrrr}\toprule",
         r"split & episodes & completed (\%) & decision steps & loop events \\\midrule"]
for sp in order:
    a = allc if sp == "all" else agg[sp]
    comp = 100.0 * a["completed"] / a["episodes"]
    lines.append(f"{sp} & {a['episodes']:,} & {comp:.1f} & {a['advance']:,} & {a['loop']:,} \\\\")
lines.append(r"\bottomrule\end{tabular}")
TAB.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))