# Item (1) addendum: FIXED-ROW-SET catalog-size sweep.
#
# Resolves the catalog-size confound in the published sweep (pub_sweep.py), which keeps
# only rows with `cur` in top-C AND `target` in top-C|stop -- i.e. the TEST ROW SET ITSELF
# changes with the budget C, so the class-vs-edge margin at different C is confounded by
# which rows survive.
#
# Here:
#   * THE ROW SET IS FIXED ONCE (every test row with a representable cur; an unseen cur
#     gets its own <unk_cur> slot) and is identical for all C and for both arms;
#   * out-of-catalog TARGETS are MASKED to an explicit <unk> class that is always offered
#     at decode (masked-renorm over the declared list), instead of dropping the row;
#   * the feature vector (full-vocabulary cur one-hot + the 9 numeric flags) is held
#     CONSTANT across C, so only the declared OUTPUT vocabulary shrinks with C --
#     exactly "hold recipe and features fixed while shrinking the option vocabulary
#     to top-C actions".
#
# Three metric views at each C, all on that one fixed row set:
#   masked_acc_unk    argmax over (concrete in-catalog options + stop + <unk>)  <-- PRIMARY
#                     the (C+2)-way question "in-catalog, or OOV? which?"; the
#                     class-vs-edge margin here is the headline and is confound-free
#   masked_acc_oov0   argmax over the concrete list only; OOV-target rows score 0
#   masked_acc_incat  conditional accuracy on rows whose target IS in top-C|stop,
#                     with incat_n -- directly comparable to the published numbers
# plus oov_target_rate and drop_frac_orig (the PUBLISHED filter's discarded fraction --
# the confound's magnitude).
#
# Writes /lw2/addendum/pub_sweep_fixed_results.json (mirrored to experiments/addendum/).
# Never touches experiments/real/*.json.
import json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn

PUB = Path("/lw2/pubdata") if Path("/lw2").exists() else None
assert PUB and PUB.exists(), "pubdata not found at /lw2/pubdata (this run is remote-only)"
OUT = Path("/lw2/addendum"); OUT.mkdir(exist_ok=True)

RUNS = 3          # parity with the published sweep (mean over 3 seeds)
EPOCHS = 45
BATCH = 512
NUMS8 = 8         # the eight numeric features + one first-flag
UNK, UNK_CUR = "<unk>", "<unk_cur>"

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, d, k, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_corpus(name):
    """Row set + feature matrix built ONCE here; never rebuilt per budget."""
    stop = json.loads((PUB / f"{name}_e2_meta.json").read_text(encoding="utf-8"))["stop"]
    rows = {sp: read_jsonl(PUB / f"{name}_e2_{sp}.jsonl") for sp in ("train", "val", "test")}
    freq = Counter(r["cur"] for r in rows["train"])
    freq.update(r["target"] for r in rows["train"] if r["target"] != stop)
    vocab = [a for a, _ in freq.most_common()]
    asort = sorted(vocab)
    amap_fixed = {a: i for i, a in enumerate(asort)}
    amap_fixed[stop] = len(asort)
    amap_fixed[UNK_CUR] = len(asort) + 1
    d = len(amap_fixed) + NUMS8 + 1          # one-hots + 8 numerics + first-flag

    c = {"stop": stop, "vocab": vocab, "d": d}
    for sp, recs in rows.items():
        X = np.zeros((len(recs), d), dtype=np.float32)
        cur, tgt, opts = [], [], []
        for i, r in enumerate(recs):
            t = r["t"]
            X[i, amap_fixed.get(r["cur"], amap_fixed[UNK_CUR])] = 1.0
            X[i, -(NUMS8 + 1):] = [t["step_idx"] / 60.0, t["prefix_len"] / 20.0,
                                   t["n_distinct"] / 20.0, t["revisit_ratio"],
                                   math.log1p(t["visit_count"]) / 3.0,
                                   t["out_deg"] / 20.0, t["out_topfreq"],
                                   t["out_entropy"] / 3.0,
                                   1.0 if t.get("first", t["step_idx"] == 0) else 0.0]
            cur.append(r["cur"]); tgt.append(r["target"]); opts.append(r["options"])
        c[sp] = {"X": X, "cur": cur, "tgt": tgt, "opts": opts}
    return c


def build_edge_vocab(cur, opts, active, stop):
    fwd = defaultdict(Counter)
    for i, r in enumerate(cur):
        if r not in active:
            continue
        for o in opts[i]:
            if o != stop and o in active and r != o:
                fwd[r][o] += 1
    ev = {}
    for a, ctr in fwd.items():
        for b in ctr:
            ev[(a, b)] = f"e{len(ev):04d}"
    return ev


def encode(sp, c, edge, mp, active, stop, ev):
    """Returns y, declared lists WITH the <unk> escape (OIS), the concrete list WITHOUT
    it (OISK), an OOV-target mask and its complement. The concrete list never contains
    <unk>, so `hit_con` on an OOV-target row is False by construction (= the oov0 view)."""
    YS, OIS, OISK, OISP, OOV, INC, POK = [], [], [], [], [], [], []
    cur, tgt, opts = c[sp]["cur"], c[sp]["tgt"], c[sp]["opts"]
    unk = mp[UNK]
    stop_id = mp["<stop_edge>"] if edge else mp[stop]
    for i in range(len(cur)):
        r, t, o = cur[i], tgt[i], opts[i]
        # PAIRED name list: identical CONCRETE candidates for BOTH arms -- subdir only the
        # options whose (cur->o) edge is train-supported (the edge arm cannot name any other),
        # plus stop. This removes the candidate-count asymmetry of the per-natural lists.
        Lp = [stop] + [x for x in o if x != stop and x in active and (r, x) in ev]
        if edge:
            def ev1(ver):
                if ver == stop:
                    return stop_id
                return mp.get(ev.get((r, ver)), unk) if ver in active else unk
            y = ev1(t)
            oov = (t != stop and t not in active) or (t in active and (r, t) not in ev)
            con = [stop_id]
            for x in o:
                if x != stop and x in active and (r, x) in ev:
                    e = mp[ev[(r, x)]]
                    if e not in con:
                        con.append(e)
        else:
            y = mp[stop] if t == stop else mp.get(t, unk)
            oov = (t != stop and t not in active)
            con = [stop_id]
            for x in o:
                if x == stop or x not in active:
                    continue
                z = mp[x]
                if z != unk and z not in con:
                    con.append(z)
        YS.append(y)
        OISK.append(list(con))
        OIS.append(list(con) + [unk])          # the <unk> escape is always offered
        OOV.append(oov); INC.append(not oov)
        if edge:
            pid = [stop_id] + [mp[ev[(r, x)]] for x in Lp if x != stop]
        else:
            pid = [mp[x] for x in Lp]
        OISP.append(pid); POK.append((t == stop) or (t in Lp))
    return (np.array(YS, dtype=np.int64), OIS, OISK, OISP,
            np.array(OOV, dtype=bool), np.array(INC, dtype=bool), np.array(POK, dtype=bool))


def fit_run(Xtr, Xte, ys_tr, ois_te, oisk_te, oisp_te, yte, ncls, seed):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1], ncls).to(DEV)
    optz = torch.optim.Adam(m.parameters(), lr=2e-3)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ys_tr, dtype=torch.long, device=DEV)
    for ep in range(EPOCHS):
        order = np.random.RandomState(seed + ep).permutation(len(ys_tr))
        for i in range(0, len(ys_tr), BATCH):
            b = order[i:i + BATCH]
            lg = m(Xt[b]); p = torch.softmax(lg, 1)
            ce = nn.functional.cross_entropy(lg, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            loss = ce + ((p - oh) ** 2).sum(1).mean()
            optz.zero_grad(); loss.backward(); optz.step()
    m.eval()
    with torch.no_grad():
        L = m(torch.tensor(Xte, device=DEV)).detach().cpu().numpy()
    hit_unk = np.zeros(len(yte), dtype=bool)
    hit_con = np.zeros(len(yte), dtype=bool)
    hit_par = np.zeros(len(yte), dtype=bool)
    for i in range(len(yte)):
        idx = np.asarray(ois_te[i]); g = L[i][idx]
        hit_unk[i] = (idx[int(np.exp(g - g.max()).argmax())] == yte[i])
        jdx = np.asarray(oisk_te[i]); h = L[i][jdx]
        hit_con[i] = (jdx[int(np.exp(h - h.max()).argmax())] == yte[i])
        pdx = np.asarray(oisp_te[i]); q = L[i][pdx]
        hit_par[i] = (pdx[int(np.exp(q - q.max()).argmax())] == yte[i])
    del m, Xt, yt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hit_unk, hit_con, hit_par


def sweep_corpus(name):
    c = load_corpus(name)
    stop, vocab = c["stop"], c["vocab"]
    budgets = sorted({B for B in (154, 512, 2048, 8192, 16384, 32768) if B <= len(vocab)}
                     | {len(vocab)})
    out = []
    for B in budgets:
        active = set(vocab[:B])
        order_a = sorted(active)
        amap_c = {a: i for i, a in enumerate(order_a)}
        amap_c[stop] = len(order_a); amap_c[UNK] = len(order_a) + 1
        ev = build_edge_vocab(c["train"]["cur"], c["train"]["opts"], active, stop)
        eids = sorted(set(ev.values()))
        emap = {e: i for i, e in enumerate(eids)}
        emap["<stop_edge>"] = len(eids); emap[UNK] = len(eids) + 1

        # confound magnitude: what the PUBLISHED filter would have kept at this B
        def kept(sp):
            cu, tg = c[sp]["cur"], c[sp]["tgt"]
            n = sum(1 for i in range(len(cu))
                    if cu[i] in active and (tg[i] in active or tg[i] == stop))
            return n / max(1, len(cu))
        drop_tr = round(1.0 - kept("train"), 4)
        drop_te = round(1.0 - kept("test"), 4)

        for edge in (False, True):
            mp = emap if edge else amap_c
            ys_tr, _, _, _, _, _, _ = encode("train", c, edge, mp, active, stop, ev)
            # SHARED support mask (identical for BOTH arms) isolates parameterization
            # quality from vocabulary sparsity: a row is shared iff the target is stop or
            # its action is in top-B AND its (cur->target) edge was seen in train.
            tg_te, cu_te = c["test"]["tgt"], c["test"]["cur"]
            shared = np.array([(t == stop) or (t in active and (cu_te[i], t) in ev)
                               for i, t in enumerate(tg_te)], dtype=bool)
            yte, ois_te, oisk_te, oisp_te, oov, inc, pok = encode("test", c, edge, mp, active, stop, ev)
            HU, HO, HC, HS, HP = [], [], [], [], []
            for seed in range(RUNS):
                hu, hc, hp = fit_run(c["train"]["X"], c["test"]["X"], ys_tr,
                                     ois_te, oisk_te, oisp_te, yte, len(mp), seed)
                HU.append(float(hu.mean()))              # diagnostic ( <unk> escape on )
                HO.append(float(hc.mean()))              # PRIMARY  oov0, no escape
                HC.append(float(hc[inc].mean()) if inc.any() else 0.0)   # conditional
                HS.append(float(hc[shared].mean()) if shared.any() else 0.0)  # shared set
                HP.append(float(hp[pok].mean()) if pok.any() else 0.0)        # strictly paired
            out.append({
                "catalog": B, "space": "edge" if edge else "action",
                "n_train_fixed": int(len(ys_tr)), "n_test_fixed": int(len(yte)),
                "n_classes": int(len(mp)),
                "masked_acc_unk": {"mean": round(float(np.mean(HU)), 4),
                                   "std": round(float(np.std(HU)), 4),
                                   "n_seeds": RUNS,
                                   "per_seed": [round(h, 4) for h in HU]},
                "masked_acc_oov0": {"mean": round(float(np.mean(HO)), 4),
                                    "std": round(float(np.std(HO)), 4),
                                    "n_seeds": RUNS,
                                    "per_seed": [round(h, 4) for h in HO]},
                "masked_acc_incat": {"mean": round(float(np.mean(HC)), 4),
                                     "std": round(float(np.std(HC)), 4),
                                     "n_seeds": RUNS,
                                     "per_seed": [round(h, 4) for h in HC]},
                "masked_acc_shared": {"mean": round(float(np.mean(HS)), 4),
                                      "std": round(float(np.std(HS)), 4),
                                      "n_seeds": RUNS,
                                      "per_seed": [round(h, 4) for h in HS]},
                "masked_acc_paired": {"mean": round(float(np.mean(HP)), 4),
                                      "std": round(float(np.std(HP)), 4),
                                      "n_seeds": RUNS,
                                      "per_seed": [round(h, 4) for h in HP]},
                "paired_n": int(pok.sum()), "paired_frac": round(float(pok.mean()), 4),
                "shared_n": int(shared.sum()),
                "incat_n": int(inc.sum()), "incat_frac": round(float(inc.mean()), 4),
                "oov_target_rate": round(float(oov.mean()), 4),
                "drop_frac_orig_train": drop_tr, "drop_frac_orig_test": drop_te,
            })
            print(f"{name} B={B:6d} {'edge  ' if edge else 'action'} "
                  f"oov0 {np.mean(HO):.4f}  shared {np.mean(HS):.4f} (n {shared.sum()})"
                  f"  incat {np.mean(HC):.4f}  oovRate {oov.mean():.3f}"
                  f"  dropOrig(te) {drop_te:.3f}  n {len(yte)}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    only, outname = None, "pub_sweep_fixed_results.json"
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--only":
            only = sys.argv[i + 1].split(",")
        elif a == "--out":
            outname = sys.argv[i + 1]
    res = {}
    names = [p.name.replace("_e2_meta.json", "") for p in sorted(PUB.glob("*_e2_meta.json"))]
    if only:
        names = [n for n in names if n in only]
    for nm in names:
        print("=== " + nm + " ===", flush=True)
        res[nm] = sweep_corpus(nm)
    # join action/edge at each budget into an explicit class-vs-edge margin
    marg = {}
    for nm, lst in res.items():
        if nm.startswith("_"):
            continue
        byB = {}
        for r in lst:
            byB.setdefault(r["catalog"], {})[r["space"]] = r
        marg[nm] = []
        for B, d in sorted(byB.items()):
            if "action" in d and "edge" in d:
                def m(k):
                    a = d["action"][k]; e = d["edge"][k]
                    a = a["mean"] if isinstance(a, dict) else a
                    e = e["mean"] if isinstance(e, dict) else e
                    return round((float(a) - float(e)) * 100.0, 2)
                marg[nm].append({"catalog": B,
                                 "margin_oov0_pts": m("masked_acc_oov0"),
                                 "margin_shared_pts": m("masked_acc_shared"),
                                 "margin_incat_pts": m("masked_acc_incat"),
                                 "margin_unk_pts": m("masked_acc_unk"),
                                 "margin_paired_pts": m("masked_acc_paired"),
                                 "drop_frac_orig_test": d["action"]["drop_frac_orig_test"]})
    res["_margins"] = marg
    res["_protocol"] = {
        "row_set": "FIXED across all budgets (every test row; unseen cur -> <unk_cur>)",
        "targets_oov": "masked to an always-offered <unk> class, never dropped",
        "features": "full-vocabulary cur one-hot + 9 numeric flags, CONSTANT across budgets",
        "primary_metric": "masked_acc_oov0 -- fixed rows, <unk> NEVER offered, out-of-support targets score 0 for BOTH arms",
        "shared_set_metric": "masked_acc_shared -- rows whose target is in top-B AND whose (cur->target) edge is in train support; identical mask for both arms, isolates parameterization quality from vocabulary sparsity",
        "diagnostic_metric": "masked_acc_unk -- with an always-offered <unk> escape; oov_target_rate differs per arm (the edge arm also calls unseen transitions OOV), so use for shape only",
        "conditional_metric": "masked_acc_incat -- accuracy on rows with an in-support target for that arm, with incat_n; closest to the published numbers",
        "margin": "_margins[*] = action - edge at each budget (percentage points, oov0/shared/incat)",
        "output_vocab": "top-B + stop + <unk> (action) / train in-catalog edge ids + <stop_edge> + <unk> (edge)",
        "epochs": EPOCHS, "batch": BATCH, "n_seeds": RUNS,
        "resolves": "the published pub_sweep.py cur/target co-filter that shrank the test set with C",
    }
    (OUT / outname).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT / outname, flush=True)