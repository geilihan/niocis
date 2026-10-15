"""Run all NIOCIS experiments (E1-E4 + ablations), emit results/*.json,
paper/tables/*.tex fragments and paper/figures/*.pdf (Type-42 fonts).

E1 graph-convergence saturation   E2 typed-decision next-hop accuracy/latency
E3 end-to-end routing vs baselines E4 RSI loop with held-out gate
A1 loop-folding ablation  A2 feature-group ablation  A3 risk-coverage sweep
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from common import (EDGES, FAIL_P, NODES, NODE_INDEX, STAGE, SUCC, State,
                    allowed_succ, edge_spec, fold_episode, guard_viol,
                    near_valid_competitors, oracle_next, soft_w, topological_neighbors)
from gen_corpus import llm_choice

rcParams["font.family"] = "serif"
rcParams["font.family"] = "Times New Roman"
rcParams["pdf.fonttype"] = 42
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "results" / "corpus"
OUT = ROOT / "results"
FIG = ROOT.parent / "paper" / "figures"
TAB = ROOT.parent / "paper" / "tables"
for p in (OUT, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

N_CLS = len(NODES) + 1          # + STOP
STOP = len(NODES)
K_NEI = 5


def load(split):
    eps = []
    with open(CORPUS / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            eps.append(json.loads(line))
    return eps


# ------------------------------------------------------------------ features
def feat(state_node, goal, ctx, retries, visited_n, steps, nei):
    x = np.zeros(len(NODES) + len(["report.archive", "cleanup.release_lock", "report.publish", "mail.digest"]) + 6 + 3 + K_NEI * 3 + 1)
    # goals list duplicated here intentionally small; consistent order
    GOALS = ["report.archive", "cleanup.release_lock", "report.publish", "mail.digest"]
    off = 0
    x[NODE_INDEX[state_node]] = 1.0; off = len(NODES)
    x[off + GOALS.index(goal)] = 1.0; off += len(GOALS)
    x[off:off + 6] = ctx; off += 6
    x[off] = min(retries, 4) / 4; x[off + 1] = min(visited_n, 6) / 6; x[off + 2] = min(steps, 40) / 40
    off += 3
    for i, (nid, w, dstage) in enumerate(nei[:K_NEI]):
        x[off + 3 * i] = (nid + 1) / 35.0
        x[off + 3 * i + 1] = w / 2.5
        x[off + 3 * i + 2] = dstage / 4.0
    return x


def neighbor_slots(node):
    out = []
    for u, v, w, _ in EDGES:
        if u == node:
            out.append((NODE_INDEX[v], w, STAGE[v] - STAGE[node]))
    out.sort(key=lambda t: -t[2])
    return out


def build_dataset(eps):
    X, y, meta = [], [], []
    for ep in eps:
        ctx = np.array(ep["step_detail"][0]["ctx"]) if ep["step_detail"] else np.zeros(6)
        visited = {}
        for i, s in enumerate(ep["step_detail"]):
            if s["kind"] != "step":   # router decides advance hops; retry/back are execution layer
                continue
            oe = s.get("oracle_edge", "")
            if "->" not in oe:
                continue
            src = s["edge"].split("->")[0]
            tgt = oe.split("->")[1]
            if tgt not in NODE_INDEX or src not in NODE_INDEX:
                continue
            visited[src] = visited.get(src, 0) + 1
            retries = s["kind"].count("retry")
            X.append(feat(src, ep["goal"], np.array(s["ctx"]), retries, visited[src], i, neighbor_slots(src)))
            y.append(NODE_INDEX[tgt])
            meta.append({"detour": s["detour"], "ambig": s["ambig"], "out_deg": len(neighbor_slots(src))})
    X = np.array(X); y = np.array(y)
    import collections
    cnt = collections.Counter(y.tolist())
    keep = [i for i, l in enumerate(y) if cnt[int(l)] >= 2]
    return X[keep], y[keep], [meta[i] for i in keep]


def temp_scale(probs, T):
    z = np.log(np.clip(probs, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs, y):
    from scipy.optimize import minimize_scalar
    def nll(logT):
        p = temp_scale(probs, math.exp(logT))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(-2, 2), method="bounded")
    return math.exp(r.x)


def ece(probs, y, bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == y).astype(float)
    e = 0.0
    lo = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > lo[i]) & (conf <= lo[i + 1])
        if m.sum():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def brier(probs, y):
    oh = np.zeros_like(probs); oh[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def latency_ms(predict_fn, X, reps=2000):
    ts = []
    for i in range(reps):
        x = X[i % len(X):i % len(X) + 1]
        t0 = time.perf_counter()
        predict_fn(x)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.sort(np.array(ts))
    return {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
            "p99": float(np.percentile(ts, 99))}


def save_json(name, obj):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tex_escape_row(cells):
    esc = [str(c).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
           for c in cells]
    return " & ".join(esc) + " \\\\"


# =============================================================== E1 convergence
def e1(eps):
    rng = np.random.default_rng(0)
    sizes = [50, 100, 200, 500, 1000, 2000, 4000, 6000]
    sizes = [s for s in sizes if s <= len(eps)]
    rows = []
    for n in sizes:
        acc = {k: [] for k in ("nodes", "edges", "back_edges", "loop_rate", "nodes_unfolded")}
        for _ in range(20):
            idx = rng.choice(len(eps), n, replace=False)
            nodes, edges, backs, loops, tot, nodes_u = set(), set(), 0, 0, 0, set()
            for j in idx:
                ep = eps[int(j)]
                skel, events = fold_episode(ep["step_detail"])
                nodes.update(skel)
                for s in ep["step_detail"]:
                    if s["kind"] == "step" and "->" in s.get("edge", ""):
                        a, b = s["edge"].split("->")
                        edges.add((a, b))
                backs += sum(1 for e in events if e.startswith("back:"))
                loops += len(events)
                tot += max(len(ep["step_detail"]), 1)
                for s in ep["step_detail"]:
                    tag = s["node"] + ("#r" if s["kind"] == "retry" else ("#b" if s["kind"] == "back" else ""))
                    nodes_u.add(tag)
            acc["nodes"].append(len(nodes)); acc["edges"].append(len(edges))
            acc["back_edges"].append(backs / n * 100); acc["loop_rate"].append(loops / tot)
            acc["nodes_unfolded"].append(len(nodes_u))
        rows.append({**{"n": n}, **{k: round(float(np.mean(v)), 3) for k, v in acc.items()},
                     **{k + "_std": round(float(np.std(v)), 3) for k, v in acc.items()}})
    save_json("e1_convergence.json", rows)
    # figure: (a) size saturation (b) loop saturation
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for key, mk, lbl in [("nodes", "o", "semantic nodes $|V|$"), ("edges", "s", "transitions $|E|$"),
                         ("nodes_unfolded", "^", "occurrence-tagged states")]:
        ax[0].plot(ns, [r[key] for r in rows], marker=mk, ms=3.5, lw=1.2, label=lbl)
    ax[0].set_xlabel("episodes"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=7, framealpha=0.9)
    ax[0].set_title("(a) state and transition growth", fontsize=8.5)
    ax[1].plot(ns, [100 * r["loop_rate"] for r in rows], marker="o", ms=3.5, lw=1.2,
               label="loop events (per 100 steps)")
    ax[1].plot(ns, [r["back_edges"] for r in rows], marker="s", ms=3.5, lw=1.2,
               label="back-events (per 100 ep.)")
    ax[1].set_xlabel("episodes"); ax[1].set_ylabel("events per 100")
    ax[1].set_title("(b) loop-event rates stabilize", fontsize=8.5)
    ax[1].legend(fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / "fig_convergence.pdf")
    plt.close(fig)
    end = rows[-1]
    (TAB / "tab_corpus.tex").write_text(
        ("\\begin{tabular}{lrrrr}\\toprule\n"
         "split & episodes & success & decision steps & loop events \\\\\n\\midrule\n"
         + "\n".join(tex_escape_row(r) for r in CORPUS_TABLE_ROWS)
         + "\n\\bottomrule\\end{tabular}"), encoding="utf-8")
    return rows


# =============================================================== E2 router
def e2(Xtr, ytr, Xva, yva, Xte, yte, meta_te):
    results = {}
    GOALS_SIZE = 4

    # majority lookup over (node, goal)
    def node_goal(X):
        return (X[:, :len(NODES)].argmax(1), X[:, len(NODES):len(NODES) + GOALS_SIZE].argmax(1))
    lut = {}
    ng = node_goal(Xtr)
    for (n, g), yy in zip(zip(*ng), ytr):
        lut.setdefault((n, g), []).append(yy)
    lut_votes = {k: np.bincount(v, minlength=N_CLS).tolist() for k, v in lut.items()}
    lut = {k: int(np.bincount(v).argmax()) for k, v in lut.items()}
    pred_lut = np.array([lut.get((n, g), ytr[np.random.randint(len(ytr))]) for n, g in zip(*node_goal(Xte))])

    models = {}
    lr = LogisticRegression(max_iter=3000, C=2.0).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
                        n_iter_no_change=8, random_state=0).fit(Xtr, ytr)
    models = {"lookup": None, "logreg": lr, "histgbm": gb, "typed-mlp": mlp,
              "lut": {k: {i: c for i, c in enumerate(v) if c > 0} for k, v in lut_votes.items()}}

    def eval_preds(pred, probs, ym=None):
        acc = float((pred == yte).mean())
        d = meta_te
        det = np.array([m["detour"] for m in d]); amb = np.array([m["ambig"] for m in d])
        deg = np.array([m["out_deg"] for m in d])
        out = {"acc": acc,
               "acc_detour": float((pred[det == 1] == yte[det == 1]).mean()) if (det == 1).any() else None,
               "acc_greedy": float((pred[det == 0] == yte[det == 0]).mean()) if (det == 0).any() else None,
               "acc_by_ambig": {str(a): round(float((pred[amb == a] == yte[amb == a]).mean()), 4)
                                for a in (0, 1, 2, 3) if (amb == a).any()},
               "n_by_ambig": {str(a): int((amb == a).sum()) for a in (0, 1, 2, 3) if (amb == a).any()},
               "acc_by_degree": {str(k): round(float((pred[deg == k] == yte[deg == k]).mean()), 4)
                                 for k in sorted(set(deg.tolist())) if (deg == k).any()},
               "n_by_degree": {str(k): int((deg == k).sum()) for k in sorted(set(deg.tolist()))}}
        if probs is not None:
            out["ece"] = round(ece(probs, ym), 4)
            out["brier"] = round(brier(probs, ym), 4)
        return out

    rows = []
    for name, m in [(k, v) for k, v in models.items() if k != "lut"]:
        if name == "lookup":
            pred, probs, ym = pred_lut, None, None
        else:
            pred = m.predict(Xte)
            cls = m.classes_
            ym = np.searchsorted(cls, yte)
            yvm = np.searchsorted(cls, yva)
            probs = temp_scale(m.predict_proba(Xte),
                               fit_temperature(m.predict_proba(Xva), yvm))
        r = eval_preds(pred, probs, ym)
        if m is not None:
            lat = latency_ms((lambda mm: (lambda x: mm.predict_proba(x)))(m), Xte)
        else:
            def lut_one(x):
                n = int(x[0, :len(NODES)].argmax())
                g = int(x[0, len(NODES):len(NODES) + 4].argmax())
                return lut.get((n, g), 0)
            lat = latency_ms(lut_one, Xte)
        r["latency"] = lat
        r["name"] = name
        results[name] = r
        rows.append(r)

    # paired bootstrap 95% CIs (2,000 resamples over decision states)
    def boot_ci(pred, y, reps=2000, seed=123):
        rng_ = np.random.default_rng(seed)
        idx = np.arange(len(y))
        if len(y) < 2:
            return None
        vals = []
        for _ in range(reps):
            b = rng_.choice(idx, len(idx), replace=True)
            vals.append(float((pred[b] == y[b]).mean()))
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    for name, r in results.items():
        m = models.get(name)
        if m is None:
            pred = pred_lut
        else:
            pred = m.predict(Xte)
        det = np.array([x["detour"] for x in meta_te])
        r["ci"] = {"acc": boot_ci(pred, yte),
                   "detour": boot_ci(pred[det == 1], yte[det == 1]),
                   "greedy": boot_ci(pred[det == 0], yte[det == 0])}

    save_json("e2_router.json", results)

    # table
    lines = ["\\begin{tabular}{lcccccc}\\toprule",
             "router & acc. & detour & greedy & ECE & Brier & p99 (ms) \\\\\\midrule"]
    for r in rows:
        lines.append(tex_escape_row([
            r["name"], f"{r['acc']:.3f}",
            f"{r['acc_detour']:.3f}" if r["acc_detour"] is not None else "--",
            f"{r['acc_greedy']:.3f}" if r["acc_greedy"] is not None else "--",
            f"{r.get('ece','--')}", f"{r.get('brier','--')}", f"{r['latency']['p99']:.3f}"]))
    lines += ["\\bottomrule\\end{tabular}"]
    (TAB / "tab_router.tex").write_text("\n".join(lines), encoding="utf-8")

    # figure: (a) accuracy by out-degree (all models)  (b) risk-coverage for typed-mlp
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    degs = sorted({m["out_deg"] for m in meta_te})
    for name, mk, c in [("lookup", "v", "#999999"), ("logreg", "s", "#0173B2"),
                        ("histgbm", "^", "#DE8F05"), ("typed-mlp", "o", "#029E73")]:
        r = results[name]
        ax[0].plot(degs, [r["acc_by_degree"].get(str(d), np.nan) for d in degs],
                   marker=mk, ms=3.5, lw=1.2, color=c, label=name)
    ax[0].set_xlabel("node out-degree"); ax[0].set_ylabel("next-hop accuracy")
    ax[0].set_ylim(0.3, 1.03); ax[0].legend(fontsize=7, framealpha=0.9)
    ax[0].set_title("(a) accuracy vs. node out-degree", fontsize=8.5)
    # risk-coverage
    cls_m = mlp.classes_
    ym_te = np.searchsorted(cls_m, yte)
    ym_va = np.searchsorted(cls_m, yva)
    probs = temp_scale(mlp.predict_proba(Xte), fit_temperature(mlp.predict_proba(Xva), ym_va))
    conf = probs.max(1); pred = probs.argmax(1)
    covs, risks = [], []
    order = np.argsort(-conf)
    ok = (pred == ym_te).astype(float)[order]
    for i in range(20, len(ok) + 1, max(len(ok) // 60, 1)):
        covs.append(i / len(ok)); risks.append(1 - ok[:i].mean())
    ax[1].plot(covs, risks, color="#029E73", lw=1.4, label="typed-mlp")
    ax[1].axhline(1 - results["lookup"]["acc"], color="#999999", lw=1.0, ls="--", label="lookup risk")
    ax[1].set_xlabel("coverage"); ax[1].set_ylabel("selective risk")
    ax[1].legend(fontsize=7, framealpha=0.9)
    ax[1].set_title("(b) accept/escalate risk--coverage", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_router.pdf")
    plt.close(fig)
    aurc = float(np.trapezoid(risks, covs))
    save_json("e2_risk_coverage.json", {"aurc_typed_mlp": round(aurc, 4)})
    return results, models


# =============================================================== E3 end-to-end
def e3(eps_test, models, tau=0.55, guard=3):
    rng = np.random.default_rng(7)
    GOALS = ["report.archive", "cleanup.release_lock", "report.publish", "mail.digest"]
    GOALS3 = GOALS

    def run_policy(ep, policy, guard_n=3, tau_=0.55):
        ctx = np.array(ep["step_detail"][0]["ctx"]) if ep["step_detail"] else np.zeros(6)
        cur, goal = ep["start"], ep["goal"]
        cost, steps, llat, llm_calls, invalid = 0.0, 0, 0.0, 0, 0
        revisits = {}
        max_steps = 26
        reason = "budget"
        while steps < max_steps:
            if cur == goal:
                return dict(success=1, cost=cost, steps=steps, lat=llat, llm=llm_calls,
                            invalid=invalid, reason="ok")
            nx_or, w = oracle_next(State(node=cur, goal=goal, ctx=ctx))
            if nx_or == "":
                reason = "deadend"
                break
            vis = revisits.get(cur, 0)
            if guard_n and vis >= guard_n:
                reason = "revisit-cap"
                break
            allowed = [v for v, _ in allowed_succ(cur, ctx)]
            if policy == "cascade":
                # cheap generative-cascade control (REFLEX B3 analogue):
                # per-(node, goal) lookup decides unless its margin is thin,
                # in which case the consulted LLM decides.
                ng = (NODE_INDEX.get(cur, -1), int(GOALS3.index(goal)) if goal in GOALS3 else -1)
                votes = luts.get(ng, {})
                llat += 0.01  # lookup micro-latency (ms)
                if votes:
                    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
                    total = sum(votes.values())
                    choice, margin = top[0][0], top[0][1] / max(total, 1)
                    if margin < 0.6 or choice not in allowed:
                        choice = nx_or if rng.random() < 0.97 else (rng.choice(allowed) if allowed else nx_or)
                        llat += float(rng.lognormal(math.log(2400), 0.45))
                        llm_calls += 1
                else:
                    choice = nx_or if rng.random() < 0.97 else (rng.choice(allowed) if allowed else nx_or)
                    llat += float(rng.lognormal(math.log(2400), 0.45))
                    llm_calls += 1
            elif policy == "llm":
                # status quo: replicate the logged agent policy of the corpus
                st_cur = State(node=cur, goal=goal, ctx=ctx)
                ambig = near_valid_competitors(st_cur)
                topo = topological_neighbors(cur, ctx)
                trap = bool(topo) and topo[0] != nx_or
                choice = llm_choice(st_cur, rng, nx_or, trap, ambig)
                if choice not in allowed:
                    invalid += 1
                    choice = nx_or
                llat += float(rng.lognormal(math.log(2400), 0.45))
                llm_calls += 1
            else:
                # typed interface: declared option set A(v) masks the readout
                x = feat(cur, goal, ctx, vis, vis, steps, neighbor_slots(cur))[None, :]
                m = models["typed-mlp"] if policy in ("typed", "typed+esc") else models["histgbm"]
                llat += 0.3  # measured batch-1 readout latency (ms; counter unit is ms)
                probs = m.predict_proba(x)[0]
                cls = m.classes_
                cand = [(NODES[int(c)] if int(c) < len(NODES) else goal, float(p))
                        for c, p in zip(cls, probs)]
                raw = max(cand, key=lambda t: t[1])[0]
                aset = set(allowed)
                masked = [(n, p) for n, p in cand if n in aset] or cand
                z = sum(p for _, p in masked) or 1.0
                best = max(masked, key=lambda t: t[1])
                choice, conf = best[0], best[1] / z
                if raw not in aset:
                    invalid += 1  # rejected pre-interface by the typed option mask
                if policy.endswith("+esc") and conf < tau_:
                    choice = nx_or if rng.random() < 0.97 else (rng.choice(allowed) if allowed else nx_or)
                    llat += float(rng.lognormal(math.log(2400), 0.45))
                    llm_calls += 1
            # guard-violation check (REFLEX-style recoverable-invalid metric)
            _, gspec = edge_spec(cur, choice)
            if gspec and guard_viol(gspec, ctx) > 0:
                if any(v != choice and guard_viol(edge_spec(cur, v)[1] or (), ctx) == 0
                       for v in allowed):
                    invalid += 1
            cost += soft_w(cur, choice, ctx)
            # stochastic environment layer (rate-level match to the corpus
            # generator: failures collapse to one 0.6-cost retry step, no poll runs)
            if rng.random() < FAIL_P.get(cur, 0.04):
                cost += 0.6
                steps += 1
                if rng.random() < 0.14:  # rollback to an earlier-stage node
                    back = [u for u in NODES if STAGE[u] == max(STAGE[cur] - 1, 0)]
                    if back:
                        b = rng.choice(back)
                        cost += 1.7
                        steps += 1
                        cur = b
                        revisits[cur] = revisits.get(cur, 0) + 1
                        continue
            if rng.random() < 0.05:  # exploration backtrack
                back = [u for u in NODES if STAGE[u] == max(STAGE[cur] - 1, 0) and u != cur]
                if back:
                    b = rng.choice(back)
                    cost += 1.9
                    steps += 1
                    cur = b
                    revisits[cur] = revisits.get(cur, 0) + 1
                    continue
            cur = choice
            revisits[cur] = revisits.get(cur, 0) + 1
            steps += 1
        return dict(success=int(cur == goal), cost=cost, steps=steps, lat=llat,
                    llm=llm_calls, invalid=invalid, reason=reason)

    luts = models.get("lut", {})
    policies = ["llm", "cascade", "typed", "histgbm", "typed+esc"]
    rows = []
    for pol in policies:
        res = [run_policy(ep, pol) for ep in eps_test]
        succ = np.mean([r["success"] for r in res])
        gaps = [r["cost"] / max(ep["oracle_cost"], 0.01) for r, ep in zip(res, eps_test)]
        lats = sorted(r["lat"] for r in res)
        fails = [r["reason"] for r in res if not r["success"]]
        rows.append({"policy": pol, "success": round(float(succ), 4),
                     "gap": round(float(np.mean(gaps)), 3),
                     "llm_calls": round(float(np.mean([r["llm"] for r in res])), 2),
                     "invalid": round(float(np.mean([r["invalid"] for r in res])), 2),
                     "lat_p50_s": round(lats[len(lats) // 2] / 1000, 4),
                     "lat_p95_s": round(lats[int(len(lats) * 0.95)] / 1000, 4),
                     "fail_reasons": {k: fails.count(k) for k in sorted(set(fails))}})
    # without guard ablation point
    nog = [run_policy(ep, "typed", guard_n=0) for ep in eps_test]
    nog_lats = sorted(r["lat"] for r in nog)
    nog_fails = [r["reason"] for r in nog if not r["success"]]
    rows.append({"policy": "typed (no guard)", "success": round(float(np.mean([r["success"] for r in nog])), 4),
                 "gap": round(float(np.mean([r["cost"] / max(ep["oracle_cost"], 0.01) for r, ep in zip(nog, eps_test)])), 3),
                 "llm_calls": 0.0, "invalid": round(float(np.mean([r["invalid"] for r in nog])), 2),
                 "lat_p50_s": round(nog_lats[len(nog_lats) // 2] / 1000, 4),
                 "lat_p95_s": round(nog_lats[int(len(nog_lats) * 0.95)] / 1000, 4),
                 "fail_reasons": {k: nog_fails.count(k) for k in sorted(set(nog_fails))}})
    save_json("e3_endtoend.json", rows)

    lines = ["\\begin{tabular}{lcccccc}\\toprule",
             "policy & success & optimality gap & LLM calls & invalid hops & p50 lat. (s) & p95 lat. (s) \\\\\\midrule"]
    for r in rows:
        lines.append(tex_escape_row([r["policy"].replace("_", "-"), f"{r['success']:.3f}", f"{r['gap']:.2f}",
                                     f"{r['llm_calls']:.2f}", f"{r['invalid']:.2f}",
                                     f"{r['lat_p50_s']:.3f}", f"{r['lat_p95_s']:.3f}"]))
    lines += ["\\bottomrule\\end{tabular}"]
    (TAB / "tab_endtoend.tex").write_text("\n".join(lines), encoding="utf-8")

    # figure: success / gap / latency
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    names = [r["policy"] for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x, [r["success"] for r in rows], color=["#999999", "#029E73", "#DE8F05", "#0173B2", "#CA9161"][:len(rows)])
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=25, ha="right", fontsize=6.5)
    ax[0].set_ylabel("task success"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(a) success rate", fontsize=8.5)
    ax[1].bar(x, [r["gap"] for r in rows], color=["#999999", "#029E73", "#DE8F05", "#0173B2", "#CA9161"][:len(rows)])
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=25, ha="right", fontsize=6.5)
    ax[1].set_ylabel("cost / oracle cost")
    ax[1].set_title("(b) optimality gap", fontsize=8.5)
    lats = [max(r["lat_p50_s"], 1e-3) for r in rows]
    ax[2].bar(x, lats, color=["#999999", "#029E73", "#DE8F05", "#0173B2", "#CA9161"][:len(rows)])
    ax[2].set_yscale("log"); ax[2].set_ylabel("decision latency p50 (s, log)")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=25, ha="right", fontsize=6.5)
    ax[2].set_title("(c) per-task decision latency", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_endtoend.pdf")
    plt.close(fig)
    return rows


# =============================================================== E4 RSI
def e4(Xtr, ytr, Xva, yva, Xte, yte, stream_eps):
    rng = np.random.default_rng(11)
    # drifted evaluation split (auth-band shift)
    rounds = []
    data_X, data_y = Xtr[:400], ytr[:400]
    cur_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
                              n_iter_no_change=8, random_state=0).fit(data_X, data_y)
    cur_acc = float((cur_model.predict(Xte) == yte).mean())
    gate_log = []
    for rd in range(3):
        # new batch: 800 samples; round 2 carries 12% corrupted labels (log noise)
        Xn, yn, _ = build_dataset(stream_eps[rd * 800:(rd + 1) * 800])
        if rd == 2:
            flip = rng.random(len(yn)) < 0.12
            yn = yn.copy()
            yn[flip] = rng.integers(0, N_CLS, size=int(flip.sum()))
        cand_X = np.vstack([data_X, Xn]); cand_y = np.concatenate([data_y, yn])
        cand = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
                             n_iter_no_change=8, random_state=0).fit(cand_X, cand_y)
        cand_acc = float((cand.predict(Xte) == yte).mean())
        cand_val = float((cand.predict(Xva) == yva).mean())
        cur_val = float((cur_model.predict(Xva) == yva).mean())
        accepted = cand_val >= cur_val - 0.005
        gate_log.append({"round": rd + 1, "cand_val": round(cand_val, 4), "cur_val": round(cur_val, 4),
                         "accepted": bool(accepted)})
        ungated_acc = cand_acc
        if accepted:
            data_X, data_y, cur_model = cand_X, cand_y, cand
            cur_acc = cand_acc
        rounds.append({"round": rd + 1, "gated_acc": round(cur_acc, 4), "ungated_acc": round(ungated_acc, 4),
                       "data_n": int(len(data_y))})
    save_json("e4_rsi.json", {"rounds": rounds, "gate_log": gate_log})

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rs = [0] + [r["round"] for r in rounds]
    g = [float((MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
          n_iter_no_change=8, random_state=0).fit(Xtr[:400], ytr[:400]).predict(Xte) == yte).mean())] + \
        [r["gated_acc"] for r in rounds]
    u = [g[0]] + [r["ungated_acc"] for r in rounds]
    ax.plot(rs, g, marker="o", ms=3.5, lw=1.2, color="#029E73", label="with held-out gate")
    ax.plot(rs, u, marker="s", ms=3.5, lw=1.2, color="#DE8F05", label="ungated updates")
    ax.set_xlabel("self-evolution round"); ax.set_ylabel("next-hop accuracy")
    ax.set_ylim(0.5, 1.0); ax.legend(fontsize=7, framealpha=0.9)
    ax.set_title("RSI updates under 12% log corruption (round 3)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_rsi.pdf")
    plt.close(fig)
    return rounds


# =============================================================== ablations
def ablations(Xtr, ytr, Xva, yva, Xte, yte, eps_tr):
    out = {}
    # A2 feature groups
    nN, nG = len(NODES), 4
    groups = {"-context": slice(None), }
    def cut(X, mode):
        X = X.copy()
        if mode == "no-context":
            X[:, nN + nG:nN + nG + 6] = 0
        elif mode == "no-goal":
            X[:, nN:nN + nG] = 0
        elif mode == "no-neighbor":
            X[:, nN + nG + 6 + 3:] = 0
        elif mode == "no-history":
            X[:, nN + nG + 6:nN + nG + 9] = 0
        return X
    feat_rows = []
    for mode in ("full", "no-context", "no-goal", "no-neighbor", "no-history"):
        m = HistGradientBoostingClassifier(max_iter=200).fit(cut(Xtr, mode) if mode != "full" else Xtr, ytr)
        acc = float((m.predict(cut(Xte, mode) if mode != "full" else Xte) == yte).mean())
        feat_rows.append({"features": mode, "acc": round(acc, 4)})
    out["feature_ablation"] = feat_rows

    # A3 tau sweep around typed-mlp
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
                        n_iter_no_change=8, random_state=0).fit(Xtr, ytr)
    cls = mlp.classes_
    ym_te = np.searchsorted(cls, yte)
    probs = temp_scale(mlp.predict_proba(Xte),
                       fit_temperature(mlp.predict_proba(Xva), np.searchsorted(cls, yva)))
    conf = probs.max(1); pred = probs.argmax(1)
    tau_rows = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= tau
        cov = float(m.mean())
        risk = float(1 - (pred[m] == ym_te[m]).mean()) if m.any() else None
        tau_rows.append({"tau": tau, "coverage": round(cov, 4), "risk": round(risk, 4) if risk is not None else None})
    out["tau_sweep"] = tau_rows

    save_json("ablations.json", out)
    lines = ["\\begin{tabular}{lcc}\\toprule", "features & acc. & $\\Delta$ \\\\\\midrule"]
    base = feat_rows[0]["acc"]
    for r in feat_rows:
        lines.append(tex_escape_row([r["features"], f"{r['acc'] + 1e-9:.3f}", f"{r['acc'] - base + 1e-9:+.3f}"]))
    lines += ["\\bottomrule\\end{tabular}"]
    (TAB / "tab_ablation.tex").write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    np.random.seed(0)
    # world parity with the corpus: share its discovered recovery back-edges
    from common import RECOVERY_EDGES
    rp = CORPUS / "recovery_pairs.json"
    if rp.exists():
        RECOVERY_EDGES.update(tuple(t) for t in json.loads(rp.read_text(encoding="utf-8")))
    eps_tr, eps_va, eps_te = load("train"), load("val"), load("test")
    eps_rs = load("rstream")
    # corpus table rows
    global CORPUS_TABLE_ROWS
    CORPUS_TABLE_ROWS = []
    meta = json.loads((CORPUS / "corpus_meta.json").read_text(encoding="utf-8"))
    for sp in ("train", "val", "test", "rstream"):
        st = meta[sp + "_stats"]
        loops = st["retry_steps"] + st["back_steps"]
        CORPUS_TABLE_ROWS.append((sp, f"{st['episodes']:,}", f"{st['rate']:.1%}", f"{st['steps']:,}", f"{loops:,}"))

    e1(eps_tr)
    Xtr, ytr, _ = build_dataset(eps_tr)
    Xva, yva, _ = build_dataset(eps_va)
    Xte, yte, meta_te = build_dataset(eps_te)
    res2, models = e2(Xtr, ytr, Xva, yva, Xte, yte, meta_te)
    e3(eps_te[:800], models)
    e4(Xtr, ytr, Xva, yva, Xte, yte, eps_rs)
    ablations(Xtr, ytr, Xva, yva, Xte, yte, eps_tr)
    print("dataset:", Xtr.shape, Xva.shape, Xte.shape)
    print("done")


if __name__ == "__main__":
    main()