# ADDENDUM (item 3): E4 multi-seed + episode-level split variant.
#
# Reproduces the FROZEN E4 recipe verbatim (same fit(): Adam 5e-3, 60 epochs, batch 512,
# RandomState(ep) shuffle; same 4x300 stream; same 12% corruption at round r=3 with the
# same "replace with a random NON-target declared option" rule; same rollback rule at
# 0.5pp) and opens exactly three knobs:
#
#   seed          torch init seed   (published run = 0)
#   corr_seed()   corruption draws  (published = 100 + r, i.e. seed-off 0)
#   split_mode    'row'   -> the published row-level permutation of VAL rows (leaks
#                            across an episode, DISCLOSED as leakage in the paper)
#                 'trace' -> whole-trace assignment, removes the intra-episode leak
#
# SHAPE-0 SELF-CHECK: run (seed=0, corr_off=0, split='row') must reproduce the frozen
# e4_rsi.json to the digit; we report that comparison explicitly. That is the recipe-parity
# receipt. Runs NEVER write into real/ -- output goes to addendum/.
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path("/lw2/runs") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
OUT = Path("/lw2/addendum") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "addendum")
OUT.mkdir(exist_ok=True)

meta = json.loads((SRC / "e2_meta.json").read_text(encoding="utf-8"))
STOP, action_vocab = meta["stop"], meta["action_vocab"]
A2I = {**action_vocab, STOP: len(action_vocab)}

tr = [json.loads(l) for l in (SRC / "e2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
va = [json.loads(l) for l in (SRC / "e2_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
te = [json.loads(l) for l in (SRC / "e2_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
Fe = np.load(SRC / "feats_laya_cls.npz")


def y_of(recs):
    return np.array([A2I[r["target"]] for r in recs])


class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)


def fit(X, y, seed=0, epochs=60):
    """verbatim from e4_rsi.py -- do NOT alter the recipe."""
    torch.manual_seed(seed)
    m = Probe(X.shape[1], len(A2I))
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(X.astype(np.float32))
    yt = torch.tensor(y, dtype=torch.long)
    for ep in range(epochs):
        order = np.random.RandomState(ep).permutation(len(y))
        for i in range(0, len(y), 512):
            b = order[i:i + 512]
            logits = m(Xt[b])
            p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            loss = ce + ((p - oh) ** 2).sum(1).mean()
            optz.zero_grad(); loss.backward(); optz.step()
    return m


def acc_masked(m, recs, X):
    m.eval()
    logits = m(torch.tensor(X.astype(np.float32))).detach().numpy()
    ok = 0
    for i, r in enumerate(recs):
        idx = [A2I[o] for o in r["options"]]
        gl = logits[i][idx]
        ok += (idx[int(np.exp(gl - gl.max()).argmax())] == A2I[r["target"]])
    return ok / len(recs)


def split_val(mode, off, want_stream=1200):
    """row  -> the published permutation (exact recipe at off=0)
       trace-> assign WHOLE epistemic groups (trace ids), so no episode straddles stream/gate."""
    rng = np.random.RandomState(7 + off)
    if mode == "row":
        perm = rng.permutation(len(va))
        va_sh = [va[p] for p in perm]
        X_sh = Fe["val"][perm]
        return va_sh[:want_stream], X_sh[:want_stream], va_sh[want_stream:], X_sh[want_stream:]
    groups = {}
    for i, r in enumerate(va):
        groups.setdefault(r["trace"], []).append(i)
    keys = [str(k) for k in rng.permutation(sorted(groups))]
    stream_i, gate_i = [], []
    for k in keys:
        tgt = stream_i if len(stream_i) < want_stream else gate_i
        tgt.extend(groups[k])
    s, g = [va[i] for i in stream_i], [va[i] for i in gate_i]
    return s, Fe["val"][stream_i], g, Fe["val"][gate_i]


def chain(split_mode, seed_off, corr_off):
    stream, Xs, gate, Xg = split_val(split_mode, seed_off)
    Xtr, ytr, Xte = Fe["train"], y_of(tr), Fe["test"]
    y_gate = y_of(gate)

    rounds = []
    cX, cy = [Xtr], [ytr]
    model = fit(np.concatenate(cX), np.concatenate(cy), seed=seed_off)
    gate_acc = acc_masked(model, gate, Xg)
    test_acc = acc_masked(model, te, Xte)
    rounds.append({"round": 0, "event": "base fit", "gate_acc": round(gate_acc, 4),
                   "test_acc": round(test_acc, 4)})
    rollbacks = 0
    B, n_used = 300, min(1200, len(stream))
    for r in range(1, 5):
        Xb, recs_b = Xs[(r - 1) * B: r * B], stream[(r - 1) * B: r * B]
        yb = y_of(recs_b)
        rate = 0.12 if r == 3 else 0.0
        applied = yb.copy()
        if rate > 0:
            rr = np.random.RandomState(100 + r + corr_off)
            idx = rr.choice(len(yb), int(round(rate * len(yb))), replace=False)
            for i in idx:
                choices = [A2I[o] for o in recs_b[i]["options"] if A2I[o] != yb[i]]
                if not choices:      # a row with a single declared option has no alternative
                    continue
                applied[i] = choices[rr.randint(len(choices))]
        ung = fit(np.concatenate(cX + [Xb]), np.concatenate(cy + [applied]), seed=seed_off)
        ung_gate = acc_masked(ung, gate, Xg)
        ung_test = acc_masked(ung, te, Xte)
        rounds.append({"round": r, "event": "ungated update" + (f" +{int(rate*100)}% corrupt" if rate else ""),
                       "gate_acc": round(ung_gate, 4), "test_acc": round(ung_test, 4)})
        if ung_gate >= gate_acc - 0.005:
            model, gate_acc = ung, ung_gate
            cX.append(Xb); cy.append(applied)
            ev = "gated accepted"
        else:
            ev = "gated ROLLBACK"; rollbacks += 1
        test_acc = acc_masked(model, te, Xte)
        rounds.append({"round": r, "event": ev, "gate_acc": round(gate_acc, 4),
                       "test_acc": round(test_acc, 4)})
    return {"split_mode": split_mode, "seed_off": seed_off, "corr_off": corr_off,
            "n_stream": len(stream), "n_gate": len(gate), "y_gate_n": int(len(y_gate)),
            "rollbacks": rollbacks, "rounds": rounds}


if __name__ == "__main__":
    runs = []
    for mode, offs in (("row", [0, 1000, 2000, 3000, 4000]),
                       ("trace", [1000, 2000, 3000, 4000, 5000])):
        for o in offs:
            r = chain(mode, o, o)
            runs.append(r)
            mine = [x["test_acc"] for x in r["rounds"] if x["event"] in ("base fit", "gated accepted")]
            print(f"{mode:5s} seedOff={o:5d} nstream={r['n_stream']} ngate={r['n_gate']} "
                  f"rollback={r['rollbacks']} gate-chain {mine} "
                  f"tee={[x['test_acc'] for x in r['rounds'] if x['event'].startswith('ungated')]}",
                  flush=True)

    # SHAPE-0 SELF-CHECK against the frozen e4_rsi.json (recipe-parity receipt)
    frozen = json.loads((SRC / "e4_rsi.json").read_text(encoding="utf-8"))
    mine0 = [x["test_acc"] for x in runs[0]["rounds"]]
    ref0 = [x["test_acc"] for x in frozen]
    parity = all(abs(a - b) < 5e-5 for a, b in zip(mine0, ref0)) and len(mine0) == len(ref0)
    gate0 = [x["gate_acc"] for x in runs[0]["rounds"]]
    gate_ref = [x["gate_acc"] for x in frozen]
    print("\nSHAPE-0 parity vs frozen e4_rsi.json:", "PASS" if parity else "MISMATCH")
    print("  mine  test:", mine0)
    print("  frozen test:", ref0)
    print("  mine  gate:", gate0)
    print("  frozen gate:", gate_ref)

    def agg(mode):
        sel = [r for r in runs if r["split_mode"] == mode]
        tail = [r["rounds"][-1]["test_acc"] for r in sel]
        base = [r["rounds"][0]["test_acc"] for r in sel]
        return {"n_runs": len(sel),
                "base_test_mean": round(float(np.mean(base)), 4), "base_test_std": round(float(np.std(base)), 4),
                "final_test_mean": round(float(np.mean(tail)), 4), "final_test_std": round(float(np.std(tail)), 4),
                "total_rollbacks": sum(r["rollbacks"] for r in sel),
                "rounds_test_mean": [round(float(np.mean([r["rounds"][i]["test_acc"] for r in sel])), 4)
                                     for i in range(len(sel[0]["rounds"]))]}

    res = {"_protocol": {
        "base": "verbatim e4_rsi.py recipe (fit 60ep Adam5e-3 batch512 RandomState(ep); 4x300 stream; 12% corrupt at r=3; rollback at 0.5pp)",
        "knobs": {"seed_off": "torch init + corruption + val-split offsets", "split_mode": "'row' (published) vs 'trace' (episode-level)"},
        "test_rows": "TEST kept clean and untouched, as in the published run",
        "self_check": "SHAPE-0 (row, seed_off=0, corr_off=0) must match the frozen e4_rsi.json digit-for-digit",
    }, "shape0_parity": parity, "shape0": {"mine_test": mine0, "frozen_test": ref0},
        "aggregate": {"row": agg("row"), "trace": agg("trace")}, "runs": runs}
    (OUT / "e4_rsi_multi.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\nwrote", OUT / "e4_rsi_multi.json")
    print(json.dumps(res["aggregate"], indent=1))