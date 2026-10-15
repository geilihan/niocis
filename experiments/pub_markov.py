# Memorized-transition baseline on the SAME masked decision evaluation as E2:
# variable-order interpolated backoff Markov over the folded node flow,
#   M0: train marginal        M1: P(a | v_t)          (= LUT-majority family)
#   M2: P(a | v_{t-1}, v_t)   M3: P(a | v_{t-2:})
# with lambda_l = weight_left * n_l/(n_l + 3) interpolation down to the
# marginal and option-gated argmax (the deployed masked-renorm gate restricts
# mass to the declared options). If the learned MLP cannot beat M3, next-hop
# structure is memorized transitions; if it does, the margin is real readout
# value. Fit on *_e2_train.jsonl rows, score *_e2_test.jsonl rows (target out
# of options counts as an error; acc-declared reported too).
#
# Row encoding mirrors pub_e2_train.build() exactly for lineage: targets and
# options bucket into the action-vocab index space with OOV -> stop index, and
# m1_raw_lut reproduces the published LUT arm (majority per t.node, integer
# set tie-break max(set(v), key=v.count), class-0 fallback) verbatim.
# Writes runs/pub_markov_results.json.
import json
from collections import defaultdict
from pathlib import Path

_HOST = Path("/lw2/pubdata")
PUB = _HOST if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
RUNS = Path("/lw2/runs") if Path("/lw2/runs").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")

SOURCES = {"soc": (RUNS / "e2_train.jsonl", RUNS / "e2_test.jsonl")}
for meta in sorted(PUB.glob("*_e2_meta.json")):
    name = meta.name.replace("_e2_meta.json", "")
    SOURCES[name] = (PUB / f"{name}_e2_train.jsonl", PUB / f"{name}_e2_test.jsonl")


def vocabs(name, meta_dir):
    mp = meta_dir / ("e2_meta.json" if name == "soc" else f"{name}_e2_meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    stop = meta["stop"]
    a2i = {**meta["action_vocab"], stop: len(meta["action_vocab"])}
    unk = a2i[stop]
    return a2i, unk


def read_rows(p, a2i, unk):
    """(trace, node, y, opts, in_opts, cur_s, tgt_s) with build() bucketing."""
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append((r["trace"], int(r["t"]["node"]),
                            a2i.get(r["target"], unk),
                            tuple(a2i.get(o, unk) for o in r["options"]),
                            bool(r.get("target_in_opts")), r["cur"], r["target"]))
    return out


def group_by_trace(rows):
    """Trace -> row list in file order (build writes per-trace in stream order)."""
    g, order = defaultdict(list), []
    for row in rows:
        if row[0] not in g:
            order.append(row[0])
        g[row[0]].append(row)
    return g, order


def contexts(seq, i):
    """(ctx3, ctx2, ctx1) = last k node values ending at decision i."""
    c1 = (seq[i][1],)
    c2 = (seq[i - 1][1],) + c1 if i >= 1 else None
    c3 = (seq[i - 2][1], seq[i - 1][1], seq[i][1]) if i >= 2 else None
    return c3, c2, c1


def fit(train_rows):
    g, _ = group_by_trace(train_rows)
    c1, c2, c3 = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), \
        defaultdict(lambda: defaultdict(int))
    marg = defaultdict(int)
    tot = 0
    align_ok = align_n = 0
    for seq in g.values():
        for i, (_, nd, y, _o, _i, cur_s, _t) in enumerate(seq):
            c1[(nd,)][y] += 1
            if i >= 1:
                prev, ptgt_s = seq[i - 1][1], seq[i - 1][6]
                c2[(prev, nd)][y] += 1
                align_n += 1
                if cur_s == ptgt_s:
                    align_ok += 1
            if i >= 2:
                c3[(seq[i - 2][1], seq[i - 1][1], nd)][y] += 1
            marg[y] += 1
            tot += 1
    # published LUT arm verbatim: majority per node over y indices,
    # integer set tie-break, class-0 fallback
    lut_v = defaultdict(list)
    for seq in g.values():
        for _, nd, y, _o, _i, _c, _t in seq:
            lut_v[nd].append(y)
    lut = {nd: max(set(v), key=v.count) for nd, v in lut_v.items()}
    return {"c1": c1, "c2": c2, "c3": c3, "marg": marg, "tot": tot, "lut": lut,
            "align_frac": (align_ok / align_n) if align_n else 0.0}


def predict_k(model, ctx, k, cands):
    """Order-k contextual prediction with lambda = weight_left*n/(n+3) backoff."""
    tables = {1: model["c1"], 2: model["c2"], 3: model["c3"]}
    p_final = {o: 0.0 for o in cands}
    weight_left = 1.0
    marg, tot = model["marg"], max(1, model["tot"])
    for order in (3, 2, 1):
        if order > k or len(ctx) < order:
            continue
        ctxo = tuple(ctx[len(ctx) - order:])
        d = tables[order].get(ctxo) or {}
        nn = sum(d.values())
        lam = weight_left * (nn / (nn + 3.0))
        if lam <= 0:
            continue
        for o in cands:
            p_final[o] += lam * (d.get(o, 0) / nn)
        weight_left -= lam
    if weight_left > 0:
        for o in cands:
            p_final[o] += weight_left * (marg.get(o, 0) / tot)
    return max(sorted(cands), key=lambda o: p_final[o])


def evaluate(model, test_rows):
    g, _ = group_by_trace(test_rows)
    stats = {f"m{k}": {"hits": 0, "decl_hits": 0, "decl_n": 0} for k in range(4)}
    lut_st = {"hits": 0, "mask_hits": 0}
    n = 0
    for seq in g.values():
        for i, (_, nd, y, opts, in_opts, _c, _t) in enumerate(seq):
            n += 1
            cands = sorted(set(opts))
            c3x, c2x, c1x = contexts(seq, i)
            ctx = c3x or c2x or c1x
            for k in range(4):
                if k == 0:
                    marg = model["marg"]
                    best = max(sorted(cands), key=lambda o: marg.get(o, 0))
                else:
                    best = predict_k(model, ctx, k, cands)
                st = stats[f"m{k}"]
                hit = best == y
                st["hits"] += int(hit)
                if in_opts:
                    st["decl_n"] += 1
                    st["decl_hits"] += int(hit)
            rp = model["lut"].get(nd, 0)
            lut_st["hits"] += int(rp == y)
            lut_st["mask_hits"] += int(rp in opts and rp == y)
    out = {}
    for k in range(4):
        st = stats[f"m{k}"]
        out[f"m{k}"] = {"masked": round(st["hits"] / max(1, n), 4),
                        "acc_declared": round(st["decl_hits"] / max(1, st["decl_n"]), 4),
                        "n": n}
    out["m1_raw_lut"] = {"test_acc": round(lut_st["hits"] / max(1, n), 4),
                         "masked_acc": round(lut_st["mask_hits"] / max(1, n), 4)}
    return out


results = {}
for name, (ptr, pte) in sorted(SOURCES.items()):
    if not ptr.exists() or not pte.exists():
        print(f"{name:14s} (skipped: splits missing)")
        continue
    if name == "soc":
        meta_dir = RUNS
    else:
        meta_dir = PUB
    a2i, unk = vocabs(name, meta_dir)
    train = read_rows(ptr, a2i, unk)
    test = read_rows(pte, a2i, unk)
    model = fit(train)
    ev = evaluate(model, test)
    ev["n_train"] = len(train)
    ev["n_test"] = len(test)
    ev["align_frac"] = round(model["align_frac"], 4)
    results[name] = ev
    print(f"{name:14s} align {ev['align_frac']:.3f} | " +
          " ".join(f"m{k}={100 * ev[f'm{k}']['masked']:.1f}" for k in range(4)) +
          f" | lut_raw={100 * ev['m1_raw_lut']['test_acc']:.1f}")

(RUNS / "pub_markov_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("wrote", RUNS / "pub_markov_results.json")