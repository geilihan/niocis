"""Regenerate paper figures and table fragments from the FROZEN result JSONs.

Never touches results/*.json. Addresses the P5 round-1 findings:
- global method->color mapping consistent across fig_router and fig_endtoend
  (plus a sixth color for the no-guard row)
- fig_convergence: legend clear of the transitions plateau; "fold events" naming;
  back-events moved to a twin axis (per-100-episodes exposure labeled)
- fig_router: risk axis 0..0.02, lookup level as off-scale text, tau=0.9 marker
- fig_endtoend: latency in ms on the y label stays seconds(log) but table gets ms
- fig_rsi: 12% *label* corruption in the title, larger canvas, annotated r3 values,
  white-edged markers
- tab_endtoend: latency columns in ms; tab_router: ECE fixed 4 decimals;
  tab_corpus: "success (%)"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_all import (FIG, TAB, build_dataset, fit_temperature,  # noqa: E402
                     load, temp_scale)

rcParams["font.family"] = "Times New Roman"
rcParams["pdf.fonttype"] = 42
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["hatch.linewidth"] = 0.8

RES = HERE / "results"

# one mapping for every method name anywhere in the paper's figures
COLOR = {
    "lookup": "#999999", "llm": "#4D4D4D",
    "logreg": "#0173B2", "cascade": "#56B4E9",
    "histgbm": "#DE8F05",
    "typed-mlp": "#029E73", "typed": "#029E73",
    "typed+esc": "#7E2B6B",
    "typed (no guard)": "#B35C00",
}
MARKER = {"lookup": "v", "logreg": "s", "histgbm": "^", "typed-mlp": "o"}

# fig_endtoend bar styling (P5 round-3): a >=20-gray-level lightness ladder
# (43/75/107/151/202/238) plus hatch variation on the three pairs that were
# indistinguishable in mono (llm vs typed+esc; typed vs no-guard; histgbm vs
# cascade). Cross-figure identity kept: typed green, histgbm orange.
BAR = {
    "llm": dict(face="#2B2B2B", edge="#000000", hatch=None),
    "cascade": dict(face="#A6D5F5", edge="#2E75B6", hatch="\\\\"),
    "typed": dict(face="#029E73", edge="#027A5B", hatch=None),
    "histgbm": dict(face="#DE8F05", edge="#8C5A10", hatch=None),
    "typed+esc": dict(face="#7E2B6B", edge="#FFFFFF", hatch="//"),
    "typed (no guard)": dict(face="#E5F4EC", edge="#027A5B", hatch="xx"),
}


def jload(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def fig_convergence():
    rows = jload("e1_convergence.json")
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for key, mk, lbl in [("nodes", "o", "semantic nodes $|V|$"),
                         ("edges", "s", "transitions $|E|$"),
                         ("nodes_unfolded", "^", "occurrence-tagged states")]:
        ax[0].plot(ns, [r[key] for r in rows], marker=mk, ms=3.5, lw=1.2, label=lbl)
    ax[0].set_xlabel("episodes"); ax[0].set_ylabel("count")
    ax[0].set_ylim(0, 240)
    ax[0].legend(fontsize=6.5, framealpha=0.9, labelspacing=0.35, borderpad=0.3, loc="upper left")
    ax[0].set_title("(a) state and transition growth", fontsize=8.5)
    ax[1].plot(ns, [100 * r["loop_rate"] for r in rows], marker="o", ms=3.5,
               lw=1.2, color="#029E73", label="fold events (per 100 steps)")
    ax[1].set_xlabel("episodes"); ax[1].set_ylabel("fold events (per 100 steps)", color="#029E73")
    ax[1].tick_params(axis="y", labelcolor="#029E73")
    ax2 = ax[1].twinx()
    ax2.plot(ns, [r["back_edges"] for r in rows], marker="s", ms=3.5, lw=1.2,
             color="#DE8F05", label="back-events (per 100 episodes)")
    ax2.set_ylabel("back-events (per 100 episodes)", color="#DE8F05")
    ax2.tick_params(axis="y", labelcolor="#DE8F05")
    ax[1].set_title("(b) loop-event rates stabilize", fontsize=8.5)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7, framealpha=0.9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_convergence.pdf")
    plt.close(fig)


def fig_router():
    e2 = jload("e2_router.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    degs = sorted(int(d) for d in e2["typed-mlp"]["acc_by_degree"])
    for name in ("lookup", "logreg", "histgbm", "typed-mlp"):
        r = e2[name]
        ax[0].plot(degs, [r["acc_by_degree"].get(str(d), np.nan) for d in degs],
                   marker=MARKER[name], ms=3.5, lw=1.2, color=COLOR[name], label=name)
    ax[0].set_xlabel("node out-degree"); ax[0].set_ylabel("next-hop accuracy")
    ax[0].set_ylim(0.3, 1.03); ax[0].legend(fontsize=7, framealpha=0.9)
    ax[0].set_title("(a) accuracy vs. node out-degree", fontsize=8.5)

    # risk-coverage recomputed exactly as run_all.e2 computes it (deterministic refit)
    Xtr, ytr, _ = build_dataset(load("train"))
    Xva, yva, _ = build_dataset(load("val"))
    Xte, yte, _ = build_dataset(load("test"))
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
                        n_iter_no_change=8, random_state=0).fit(Xtr, ytr)
    cls_m = mlp.classes_
    ym_te = np.searchsorted(cls_m, yte); ym_va = np.searchsorted(cls_m, yva)
    probs = temp_scale(mlp.predict_proba(Xte), fit_temperature(mlp.predict_proba(Xva), ym_va))
    conf = probs.max(1); pred = probs.argmax(1)
    covs, risks = [], []
    order = np.argsort(-conf)
    ok = (pred == ym_te).astype(float)[order]
    for i in range(20, len(ok) + 1, max(len(ok) // 60, 1)):
        covs.append(i / len(ok)); risks.append(1 - ok[:i].mean())
    aurc = float(np.trapezoid(risks, covs))
    frozen = jload("e2_risk_coverage.json")["aurc_typed_mlp"]
    assert abs(round(aurc, 4) - frozen) <= 1e-4, f"aurc drift {aurc:.5f} vs frozen {frozen}"

    ax[1].plot(covs, risks, color=COLOR["typed-mlp"], lw=1.4, label="typed-mlp", zorder=3)
    lk = 1 - e2["lookup"]["acc"]
    ax[1].text(0.30, 0.0155, f"lookup risk {lk * 100:.1f}% (off scale)",
               fontsize=7, color="#333333")
    # exact tau=0.9 operating point from ablations.json (NOT the nearest sweep
    # vertex) so the star matches the callout and caption values (R7 P5 finding)
    ab = jload("ablations.json")
    t9 = next(r for r in ab["tau_sweep"] if abs(r["tau"] - 0.9) < 1e-9)
    ax[1].plot([t9["coverage"]], [t9["risk"]], marker="*", ms=7, color="#4D4D4D", zorder=4)
    ax[1].annotate("$\\tau{=}0.9$: cov. 95.7%", (t9["coverage"], t9["risk"]),
                  textcoords="offset points", xytext=(-6, 9), fontsize=7, ha="right")
    ax[1].set_xlabel("coverage"); ax[1].set_ylabel("selective risk")
    ax[1].set_ylim(0, 0.02); ax[1].legend(fontsize=7, framealpha=0.9, loc="upper left")
    ax[1].set_title("(b) accept/escalate risk--coverage", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_router.pdf")
    plt.close(fig)


def fig_endtoend():
    rows = jload("e3_endtoend.json")
    names = [r["policy"] for r in rows]
    labels = names
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))

    def draw(a, vals):
        for xi, v, n in zip(x, vals, names):
            st = BAR[n]
            a.bar(xi, v, width=0.7, facecolor=st["face"], edgecolor=st["edge"],
                  hatch=st["hatch"], linewidth=1.0)

    for a, key, ylab in [(ax[0], "success", "task success"),
                         (ax[1], "gap", "cost / oracle cost")]:
        draw(a, [r[key] for r in rows])
        a.set_xticks(x); a.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        a.set_ylabel(ylab)
    ax[0].set_ylim(0, 1.05); ax[0].set_title("(a) success rate", fontsize=8.5)
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set_title("(b) optimality gap", fontsize=8.5)
    draw(ax[2], [max(r["lat_p50_s"] * 1000, 1.0) for r in rows])
    ax[2].set_yscale("log"); ax[2].set_ylabel("decision latency p50 (ms, log)")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax[2].set_title("(c) per-task decision latency, p50", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_endtoend.pdf")
    plt.close(fig)


def fig_rsi():
    e4 = jload("e4_rsi.json")
    rs = [0] + [r["round"] for r in e4["rounds"]]
    g = [e4["seed_test_acc"]] + [r["gated_acc"] for r in e4["rounds"]]
    u = [e4["seed_test_acc"]] + [r["ungated_acc"] for r in e4["rounds"]]
    # drawn at the size it is included at (~0.55\linewidth = 3.0in) so every
    # glyph keeps >=7pt after scaling (P5 round-3 legibility finding)
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot(rs, g, marker="o", ms=5, lw=1.5, color=COLOR["typed-mlp"],
            markeredgecolor="white", markeredgewidth=0.8, zorder=3, label="with held-out gate")
    ax.plot(rs, u, marker="s", ms=5, lw=1.5, color="#DE8F05",
            markeredgecolor="white", markeredgewidth=0.8, zorder=2, label="ungated updates")
    ax.annotate(f"{g[-1] * 100:.1f}%", (rs[-1], g[-1]), textcoords="offset points",
                xytext=(-27, 5), fontsize=7.5, color=COLOR["typed-mlp"])
    ax.annotate(f"{u[-1] * 100:.1f}%", (rs[-1], u[-1]), textcoords="offset points",
                xytext=(-27, -11), fontsize=7.5, color="#8c5a10")
    ax.set_xlabel("self-evolution round", fontsize=9)
    ax.set_ylabel("next-hop accuracy", fontsize=9, color="black")
    ax.tick_params(labelsize=8)
    ax.set_xticks(rs); ax.set_ylim(0.79, 1.02)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="lower right")
    ax.set_title("RSI updates under 12% label corruption\n(round 3)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_rsi.pdf")
    plt.close(fig)


def tables():
    # tab_router: ECE at fixed 4 decimals (values unchanged)
    e2 = jload("e2_router.json")
    lines = ["\\begin{tabular}{lrrrrrr}\\toprule",
             "router & acc. & detour & greedy & ECE & Brier & p99 (ms) \\\\\\midrule"]
    for name in ("lookup", "logreg", "histgbm", "typed-mlp"):
        r = e2[name]
        row = [name, f"{r['acc']:.3f}",
               f"{r['acc_detour']:.3f}" if r["acc_detour"] is not None else "--",
               f"{r['acc_greedy']:.3f}" if r["acc_greedy"] is not None else "--",
               f"{r['ece']:.4f}" if r.get("ece") is not None else "--",
               f"{r['brier']:.4f}" if r.get("brier") is not None else "--",
               f"{r['latency']['p99']:.3f}"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}"]
    (TAB / "tab_router.tex").write_text("\n".join(lines), encoding="utf-8")

    # tab_endtoend: latency columns in ms
    rows = jload("e3_endtoend.json")
    lines = ["\\begin{tabular}{lrrrrrr}\\toprule",
             "policy & success & optimality gap & LLM calls & invalid hops & p50 lat. (ms) & p95 lat. (ms) \\\\\\midrule"]
    for r in rows:
        row = [r["policy"], f"{r['success']:.3f}", f"{r['gap']:.2f}",
               f"{r['llm_calls']:.2f}", f"{r['invalid']:.2f}",
               f"{r['lat_p50_s'] * 1000:,.1f}", f"{r['lat_p95_s'] * 1000:,.1f}"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}"]
    (TAB / "tab_endtoend.tex").write_text("\n".join(lines), encoding="utf-8")

    # tab_corpus: explicit percent unit in the header
    cm = json.loads((RES / "corpus" / "corpus_meta.json").read_text(encoding="utf-8"))
    lines = ["\\begin{tabular}{lrrrr}\\toprule",
             "split & episodes & success (\\%) & decision steps & loop events \\\\\\midrule"]
    for sp in ("train", "val", "test", "rstream"):
        st = cm[sp + "_stats"]
        loops = st["retry_steps"] + st["back_steps"]
        # decision steps = advance hops (router decision states); the loop
        # events column (retry+back) is excluded from it (R4 P2 finding)
        adv = st["steps"] - loops
        row = [sp, f"{st['episodes']:,}", f"{st['rate'] * 100:.1f}", f"{adv:,}", f"{loops:,}"]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}"]
    (TAB / "tab_corpus.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    fig_convergence()
    fig_router()
    fig_endtoend()
    fig_rsi()
    tables()
    print("figures and tables regenerated from frozen results")


if __name__ == "__main__":
    main()