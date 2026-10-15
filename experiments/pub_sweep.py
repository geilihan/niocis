# Item #4: catalog-size sweep -- tests the paper's §7 structural-ID hypothesis
# (per-edge IDs should pay where catalogs are large). For each corpus and each
# catalog budget C (top-C TRAIN-frequency actions), score only in-catalog
# decisions (cur and target in catalog or stop) and compare class-space vs
# edge-ID MLP readouts (CE+Brier, 3 seeds), with the option graph and edge
# vocab rebuilt inside the catalog. Writes runs/pub_sweep_results.json.
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container
_host = Path("/lw2/pubdata")
PUB = _host if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
RUNS = Path("/lw2/runs") if Path("/lw2/runs").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NUMS = ["step_idx", "prefix_len", "n_distinct", "revisit_ratio", "visit_count",
        "out_deg", "out_topfreq", "out_entropy"]


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class MLP(nn.Module):
    def __init__(self, d, k, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def sweep_corpus(name):
    rows = {sp: read_jsonl(PUB / f"{name}_e2_{sp}.jsonl") for sp in ("train", "val", "test")}
    freq = Counter(r["cur"] for r in rows["train"])
    freq.update(r["target"] for r in rows["train"]
                if r["target"] != json.loads((PUB / f"{name}_e2_meta.json")
                                             .read_text(encoding="utf-8"))["stop"])
    vocab = [a for a, _ in freq.most_common()]
    stop = json.loads((PUB / f"{name}_e2_meta.json").read_text(encoding="utf-8"))["stop"]
    budgets = sorted({C for C in (154, 512, 2048, 8192, 16384, 32768) if C <= len(vocab)}
                     | {len(vocab)})
    out = []
    for C in budgets:
        active = set(vocab[:C])
        sel = {}
        for sp, recs in rows.items():
            sel[sp] = [r for r in recs if r["cur"] in active and
                       (r["target"] in active or r["target"] == stop)]
        # in-catalog option graph + edge vocab from TRAIN rows (verbatim build rule)
        train_fwd = defaultdict(Counter)
        for r in sel["train"]:
            for o in r["options"]:
                if o != stop and o in active and r["cur"] != o:
                    train_fwd[r["cur"]][o] += 1
        edge_vocab = {}
        for a, ctr in train_fwd.items():
            for b in ctr:
                edge_vocab[f"{a}->{b}"] = f"e{len(edge_vocab):04d}"
        amap = {a: i for i, a in enumerate(sorted(active))}
        amap[stop] = len(amap)
        eids = sorted(set(edge_vocab.values()))
        emap = {e: i for i, e in enumerate(eids)}
        emap["<stop_edge>"] = len(emap); emap["<unk_edge>"] = len(emap) + 1

        def rebuild(recs, edge):
            Xs, ys, opts, insp = [], [], [], []
            for r in recs:
                t = r["t"]
                x = np.zeros(len(amap) + len(NUMS) + 2, dtype=np.float32)  # goal dropped (1 agent)
                x[amap.get(r["cur"], 0)] = 1.0
                b = len(amap)
                vals = [t["step_idx"] / 60.0, t["prefix_len"] / 20.0, t["n_distinct"] / 20.0,
                        t["revisit_ratio"], math.log1p(t["visit_count"]) / 3.0,
                        t["out_deg"] / 20.0, t["out_topfreq"], t["out_entropy"] / 3.0,
                        1.0 if t.get("first", t["step_idx"] == 0) else 0.0]
                x[b:b + len(vals)] = vals
                o = [o for o in r["options"] if o in active or o == stop] + []
                if stop not in o:
                    o = o + [stop]
                if edge:
                    oi = [emap.get(edge_vocab.get(f"{r['cur']}->{oo}", "<unk_edge>"),
                                   emap["<unk_edge>"]) if oo != stop else emap["<stop_edge>"] for oo in o]
                    y = emap.get(edge_vocab.get(f"{r['cur']}->{r['target']}", "<unk_edge>"),
                                 emap["<unk_edge>"]) if r["target"] != stop else emap["<stop_edge>"]
                else:
                    oi = [amap.get(oo, amap[stop]) for oo in o]
                    y = amap.get(r["target"], amap[stop])
                Xs.append(x); ys.append(y); opts.append(oi); insp.append(r["target"] in o)
            return np.stack(Xs), np.array(ys), opts, np.array(insp)

        for edge in (False, True):
            Xtr, ytr, _, _ = rebuild(sel["train"], edge)
            Xte, yte, ote, ite = rebuild(sel["test"], edge)
            if len(Xtr) < 500 or len(Xte) < 50:
                continue
            nc = len(emap) if edge else len(amap)
            hits = []
            for seed in range(3):
                torch.manual_seed(seed)
                m = MLP(Xtr.shape[1], nc).to(DEV)
                optz = torch.optim.Adam(m.parameters(), lr=2e-3)
                Xt = torch.tensor(Xtr, device=DEV)
                yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
                for ep in range(45):
                    order = np.random.RandomState(seed + ep).permutation(len(ytr))
                    for i in range(0, len(ytr), 512):
                        b2 = order[i:i + 512]
                        lg = m(Xt[b2]); p = torch.softmax(lg, 1)
                        ce = nn.functional.cross_entropy(lg, yt[b2])
                        oh = torch.zeros_like(p).scatter_(1, yt[b2].unsqueeze(1), 1.0)
                        loss = ce + ((p - oh) ** 2).sum(1).mean()
                        optz.zero_grad(); loss.backward(); optz.step()
                m.eval()
                lg = m(torch.tensor(Xte, device=DEV)).detach().cpu().numpy()
                ok = 0
                for i in range(len(yte)):
                    g = lg[i][ote[i]]; g = g - g.max()
                    ok += (ote[i][int(np.exp(g).argmax())] == yte[i])
                hits.append(ok / len(yte))
            out.append({"catalog": C, "space": "edge" if edge else "action",
                        "n_train": len(Xtr), "n_test": len(Xte),
                        "in_space_rate": round(float(ite.mean()), 4),
                        "masked_acc": {"mean": round(float(np.mean(hits)), 4),
                                       "std": round(float(np.std(hits)), 4)}})
            print(f"{name} C={C:6d} {'edge  ' if edge else 'action'} "
                  f"acc {np.mean(hits):.4f} (n_test {len(Xte)})")
    return out


res = {}
for p in sorted(PUB.glob("*_e2_meta.json")):
    name = p.name.replace("_e2_meta.json", "")
    res[name] = sweep_corpus(name)
(RUNS / "pub_sweep_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("wrote", RUNS / "pub_sweep_results.json")