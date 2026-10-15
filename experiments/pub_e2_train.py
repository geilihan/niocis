# Readout arms on PUBLIC decision sets (per corpus). Verbatim adaptation of
# e2_train_torch.py rules (vec dims fixed on the ACTION vocab in both spaces,
# masked gate, CE vs CE+Brier) + LUT/logreg reference arms. GPU when
# available. Writes {name}_e2_results.json.
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container
_host = Path("/lw2/pubdata")
PUB = _host if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
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


def run_corpus(name):
    meta = json.loads((PUB / f"{name}_e2_meta.json").read_text(encoding="utf-8"))
    STOP = meta["stop"]
    A2I = {**meta["action_vocab"], STOP: len(meta["action_vocab"])}
    _eids = sorted(set(meta["edge_vocab"].values()))
    E2I = {e: i for i, e in enumerate(_eids)}
    E2I["<stop_edge>"] = len(E2I)
    E2I["<unk_edge>"] = len(E2I) + 1
    n_agent = len(meta["agent_vocab"])
    splits = {sp: read_jsonl(PUB / f"{name}_e2_{sp}.jsonl") for sp in ("train", "val", "test")}
    for recs in splits.values():
        for r in recs:
            r["t"]["first"] = (r["t"]["step_idx"] == 0)

    def vec(r):
        # dims fixed on the ACTION vocab in both spaces (verbatim from SOC)
        t = r["t"]
        x = np.zeros(len(A2I) + n_agent + len(NUMS) + 1, dtype=np.float32)
        x[t["node"]] = 1.0
        x[len(A2I) + t["goal"]] = 1.0
        b = len(A2I) + n_agent
        vals = [t["step_idx"] / 60.0, t["prefix_len"] / 20.0, t["n_distinct"] / 20.0,
                t["revisit_ratio"], math.log1p(t["visit_count"]) / 3.0,
                t["out_deg"] / 20.0, t["out_topfreq"], t["out_entropy"] / 3.0,
                1.0 if t["first"] else 0.0]
        x[b:b + len(vals)] = vals
        return x

    def build(recs, edge_target):
        vocab, unk = (E2I, E2I["<unk_edge>"]) if edge_target else (A2I, A2I[STOP])
        X = np.stack([vec(r) for r in recs])
        y = np.array([vocab.get(r["target_edge_id"] if edge_target else r["target"], unk) for r in recs])
        opts = [[vocab.get(o, unk) for o in (r["option_edge_ids"] if edge_target else r["options"])]
                for r in recs]
        insp = np.array([r["target_in_opts"] for r in recs])
        return X, y, opts, insp

    def evaluate(logits, y, opts, insp):
        confs, cors, hit, hitd, nd = [], [], 0, 0, 0
        for i in range(len(y)):
            gl = logits[i][opts[i]]
            gl = gl - gl.max()
            p = np.exp(gl); p /= p.sum()
            j = int(p.argmax())
            ok = (opts[i][j] == y[i])
            confs.append(float(p[j])); cors.append(ok)
            hit += ok
            if insp[i]:
                hitd += ok; nd += 1
        order = sorted(range(len(confs)), key=lambda i: -confs[i])
        run, curve = 0, []
        for k, i in enumerate(order, 1):
            run += (not cors[i])
            if k % max(1, len(order) // 20) == 0:
                curve.append({"coverage": round(k / len(order), 4), "risk": round(run / k, 4)})
        # NOTE (R144): this quantity is the mean absolute calibration gap
        #   mean_i |conf_i - correct_i|, NOT the 15-bin ECE declared in the paper's
        #   method section (the binned estimator lives in e2_train_torch.py and
        #   e2_laya_probe.py). The key stays "ece15" only so the frozen
        #   pub_*_e2_results.json schema keeps round-tripping; no number in the
        #   manuscript is taken from this field (see REPRODUCE.md, section 5).
        calib_gap = sum(abs(c - o) for c, o in zip(confs, cors)) / len(confs)
        aurc = sum(pt["risk"] for pt in curve) / max(1, len(curve))
        return {"masked_acc": round(hit / len(y), 4), "acc_declared": round(hitd / max(1, nd), 4),
                "n_in_space": int(nd), "ece15": round(calib_gap, 4), "aurc": round(aurc, 4),
                "curve": curve[:21]}

    def train_model(Xtr, ytr, n_classes, objective, seed, epochs=60):
        torch.manual_seed(seed)
        m = MLP(Xtr.shape[1], n_classes).to(DEV)
        optz = torch.optim.Adam(m.parameters(), lr=2e-3)
        Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, dtype=torch.long, device=DEV)
        n = len(Xtr)
        for ep in range(epochs):
            order = np.random.RandomState(seed + ep).permutation(n)
            for i in range(0, n, 512):
                b = order[i:i + 512]
                lg = m(Xt[b]); p = torch.softmax(lg, 1)
                ce = nn.functional.cross_entropy(lg, yt[b])
                oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
                loss = ce if objective == "ce" else ce + ((p - oh) ** 2).sum(1).mean()
                optz.zero_grad(); loss.backward(); optz.step()
        return m

    res = {"corpus": name, "device": DEV, "n_train": len(splits["train"]),
           "n_test": len(splits["test"])}
    for edge in (False, True):
        tag = "edge" if edge else "action"
        Xtr, ytr, otr, itr = build(splits["train"], edge)
        Xte, yte, ote, ite = build(splits["test"], edge)
        n_classes = len(E2I) if edge else len(A2I)
        if not edge:
            maj = {}
            for nd, yy in zip([r["t"]["node"] for r in splits["train"]], ytr):
                maj.setdefault(nd, []).append(int(yy))
            maj = {nd: max(set(v), key=v.count) for nd, v in maj.items()}
            pred = [maj.get(r["t"]["node"], 0) for r in splits["test"]]
            res["action_lut_majority"] = {
                "test_acc": round(float(np.mean([p == y for p, y in zip(pred, yte)])), 4),
                "masked_acc": round(float(np.mean([p in o and p == y for p, o, y in zip(pred, ote, yte)])), 4)}
            lr = LogisticRegression(max_iter=800, random_state=0).fit(Xtr, ytr)
            pred = lr.predict(Xte)
            res["action_logreg"] = {
                "test_acc": round(float(np.mean([p == y for p, y in zip(pred, yte)])), 4),
                "masked_acc": round(float(np.mean([p in o and p == y for p, o, y in zip(pred, ote, yte)])), 4)}
        for obj in ("ce", "cebrier"):
            mets = []
            for seed in range(5):
                m = train_model(Xtr, ytr, n_classes, obj, seed).eval()
                lg = m(torch.tensor(Xte, device=DEV)).detach().cpu().numpy()
                mets.append(evaluate(lg, yte, ote, ite))
            res[f"{tag}_mlp_{obj}"] = {
                "masked_acc": {"mean": round(float(np.mean([m["masked_acc"] for m in mets])), 4),
                               "std": round(float(np.std([m["masked_acc"] for m in mets])), 4)},
                "acc_declared": round(float(np.mean([m["acc_declared"] for m in mets])), 4),
                "ece15": round(float(np.mean([m["ece15"] for m in mets])), 4),
                "per_seed": [{kk: m[kk] for kk in ("masked_acc", "acc_declared", "ece15")} for m in mets]}
    (PUB / f"{name}_e2_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(name, "action cebrier", res["action_mlp_cebrier"]["masked_acc"],
          "edge cebrier", res["edge_mlp_cebrier"]["masked_acc"])


print("device:", DEV)
for p in sorted(PUB.glob("*_e2_meta.json")):
    name = p.name.replace("_e2_meta.json", "")
    if (PUB / f"{name}_e2_results.json").exists():
        print(f"[skip] {name}: results exist (resume mode)")
        continue
    run_corpus(name)