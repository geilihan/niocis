# Real-data figures + tables. Reads ONLY experiments/real/*.json (frozen artifacts).
# Each builder self-guards on its JSON; rerun after pending runs land to fill in rows.
import os
import json
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HOST = Path("/lw2/runs")
_PAPER = Path(__file__).resolve().parent.parent / "paper"
REAL = _HOST if _HOST.exists() else Path(__file__).resolve().parent / "real"
FIG = (_HOST / "figures") if _HOST.exists() else _PAPER / "figures"
TAB = (_HOST / "tables") if _HOST.exists() else _PAPER / "tables"

LADDER = ["#2B2B2B", "#4D4D4D", "#6A6A6A", "#979797", "#C6C6C6", "#EFEFEF"]
ACCENT = "#029E73"
ACCENT2 = "#2E75B6"
RED = "#DE8F05"
PURPLE = "#7E2B6B"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})


def jload(name):
    p = REAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", name)


def save_tab(lines, name):
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------- fig 1
def fig_convergence():
    d = jload("e1_saturation.json")
    ks = sorted(int(k) for k in d["curve"])
    q = lambda key, stat: [d["curve"][str(k)][key][stat] for k in ks]
    fig, ax = plt.subplots(figsize=(3.3, 2.2), nrows=1, ncols=2, gridspec_kw={"wspace": 0.72})
    a0 = ax[0]
    a0.plot(ks, q("nodes", "p50"), color=ACCENT, lw=1.6, marker="o", ms=2.5, label="semantic nodes")
    a0.plot(ks, q("fwd", "p50"), color=LADDER[0], lw=1.6, marker="s", ms=2.5, label="distinct pairs ($E^{+}\\cup E^{-}$)")
    a0.plot(ks, q("back", "p50"), color=RED, lw=1.6, marker="^", ms=2.5, label="back-edge pairs (distinct, $E^-$)")
    a0.fill_between(ks, q("nodes", "min"), q("nodes", "max"), color=ACCENT, alpha=0.15, lw=0)
    a0.fill_between(ks, q("fwd", "min"), q("fwd", "max"), color=LADDER[0], alpha=0.12, lw=0)
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_xlabel("episodes (log)")
    a0.set_ylabel("count (log)")
    a0.set_title("(a) graph saturation", fontsize=9)
    h, labels = a0.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.32)
    fig.legend(h, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.025),
               fontsize=6, handlelength=0.9, labelspacing=0.28, columnspacing=0.8)
    a1 = ax[1]
    a1.plot(ks, q("loops_per_100_steps", "p50"), color=ACCENT2, lw=1.6, marker="o", ms=2.5)
    a1.set_xlabel("episodes (log)")
    a1.set_ylabel("loop events / 100 steps", color=ACCENT2, labelpad=1)
    a1.set_xscale("log")
    a1.tick_params(axis="y", colors=ACCENT2)
    a1.set_title("(b) loop-event rates", fontsize=9)
    a2 = a1.twinx()
    a2.plot(ks, q("back_per_100_eps", "p50"), color=RED, lw=1.6, marker="^", ms=2.5, ls="--")
    a2.set_ylabel("back events / 100 episodes", color=RED)
    a2.tick_params(axis="y", colors=RED)
    save(fig, "fig_convergence")


# ---------------------------------------------------------------- fig 2 + tab router
def router_rows():
    rows = []
    e2 = jload("e2_results.json")
    if e2:
        def put(arm, acc, decl, ece, aurc, note="", acc_disp=None, decl_disp=None):
            rows.append(dict(arm=arm, acc=acc, decl=decl, ece=ece, aurc=aurc, note=note,
                             acc_disp=acc_disp, decl_disp=decl_disp))
        put("majority arm", e2["action_lut_majority"]["test_acc"], None, None, None,
            acc_disp=pct1_from_rate(e2["action_lut_majority"]["test_acc"], 3875))
        put("logistic regression", e2["action_logreg"]["test_acc"], None, None, None,
            acc_disp=pct1_from_rate(e2["action_logreg"]["test_acc"], 3875))
        for arm, key in (("feature MLP, CE", "action_mlp_ce"),
                         ("feature MLP, CE$+$Brier", "action_mlp_cebrier"),
                         ("edge-ID readout, CE", "edge_mlp_ce"),
                         ("edge-ID readout, CE$+$Brier", "edge_mlp_cebrier")):
            v = e2[key]
            e2_ps = v['per_seed']
            put(arm, v["masked_acc"]["mean"], v.get("acc_declared", {}).get("mean"),
                v["ece15"]["mean"], v["aurc"]["mean"],
                note=f"$\\pm {v['masked_acc']['std']:.3f}$",
                acc_disp=pct1_from_perseed([s2['masked_acc'] for s2 in e2_ps], 3875),
                decl_disp=(pct1_from_perseed([s2['acc_declared'] for s2 in e2_ps], 3568)
                           if v.get('acc_declared') else None))
    pr = jload("e2_layaprobe_results.json")
    if pr:
        for arm, key in (("hidden-state probe, CE", "action_layaprobe_ce"),
                         ("hidden-state probe, CE$+$Brier", "action_layaprobe_cebrier"),
                         ("hidden-state probe (edge-ID), CE$+$Brier", "edge_layaprobe_cebrier")):
            v = pr[key]
            pr_ps = v['per_seed']
            put(arm, v["masked_acc"]["mean"], v.get("acc_declared", {}).get("mean"),
                v["ece15"]["mean"], None, note=f"$\\pm {v['masked_acc']['std']:.3f}$",
                acc_disp=pct1_from_perseed([s2['masked_acc'] for s2 in pr_ps], 3875),
                decl_disp=(pct1_from_perseed([s2['acc_declared'] for s2 in pr_ps], 3568)
                           if v.get('acc_declared') else None))
    zs = jload("e2_laya_results.json")
    if zs:
        v = zs["test"]
        # e2_laya_results reports the in-space rate; every other row's acc column
        # counts out-of-space targets as errors, so rescore zero-shot over all test
        # decisions from its frozen per-decision probs and disclose both numbers.
        acc_all = v["acc"]
        acc_all_disp = None
        raw = jload("e2_laya_raw.json")
        if raw and raw.get("test"):
            recs = raw["test"]
            hits = sum(1 for r in recs if max(r["probs"], key=r["probs"].get) == r["target"])
            acc_all = hits / len(recs)
            acc_all_disp = pct1_from_counts(hits, len(recs))
        put("Laya zero-shot typed choice", acc_all, v["acc"], v["ece15"], None,
            note=f"$T={zs['temperature']}$", acc_disp=acc_all_disp,
            decl_disp=pct1_from_rate(v['acc'], v['n_eval']) if v.get('n_eval') else None)
    return rows


def tab_router():
    rows = router_rows()
    if not rows:
        return
    lines = [r"{\setlength{\tabcolsep}{4.5pt}\begin{tabular}{lrrrrl}\toprule",
             r"readout arm & acc.\ (\%) & in-space (\%) & ECE-15 & AURC & $\pm$ std / config \\\midrule"]
    # Arms whose paired interval against the indicated reference includes zero
    # (main_body.tex tab:router caption) -- regenerating must not drop these marks.
    DAGGER = {"logistic regression", "feature MLP, CE", "feature MLP, CE$+$Brier",
              "edge-ID readout, CE", "edge-ID readout, CE$+$Brier"}
    for r in rows:
        arm = r["arm"] + (r"$^{\dagger}$" if r["arm"] in DAGGER else "")
        acc = r.get("acc_disp") or f"{100 * r['acc']:.1f}"
        decl = (r.get("decl_disp") or f"{100 * r['decl']:.1f}") if r["decl"] is not None else "--"
        ece = f"{r['ece']:.3f}" if r["ece"] is not None else "--"
        aurc = f"{r['aurc']:.3f}" if r["aurc"] is not None else "--"
        lines.append(f"{arm} & {acc} & {decl} & {ece} & {aurc} & {r['note'] or '--'} \\\\")
    lines.append(r"\bottomrule\end{tabular}}")
    save_tab(lines, "tab_router")


def fig_router():
    rows = router_rows()
    if not rows:
        return
    e2 = jload("e2_results.json")
    fig, ax = plt.subplots(figsize=(3.4, 2.5), nrows=1, ncols=2, gridspec_kw={"wspace": 0.8})
    SHORT = {"majority arm": "maj. arm", "logistic regression": "logreg",
             "feature MLP, CE": "MLP CE", "feature MLP, CE$+$Brier": "MLP CE+Brier",
             "edge-ID readout, CE": "edge-ID CE", "edge-ID readout, CE$+$Brier": "edge-ID CE+Brier",
             "hidden-state probe, CE": "probe CE", "hidden-state probe, CE+$Brier": "probe CE+Brier",
             "hidden-state probe (edge-ID), CE$+$Brier": "probe edge-ID",
             "Laya zero-shot typed choice": "zero-shot"}
    names = [SHORT.get(r["arm"], r["arm"]) for r in rows]
    accs = [100 * r["acc"] for r in rows]
    colors = [LADDER[0], LADDER[2], ACCENT, ACCENT, RED, RED, PURPLE, PURPLE,
              PURPLE, LADDER[4]]
    a0 = ax[0]
    y = np.arange(len(names))[::-1]
    a0.barh(y, accs, color=colors[: len(names)], edgecolor="black", lw=0.4)
    a0.set_yticks(y)
    a0.set_yticklabels(names, fontsize=6)
    a0.set_xlabel("test acc. (%)")
    a0.set_title("(a) readout arms", fontsize=9)
    a1 = ax[1]
    if e2:
        for key, lbl, c, mk, ls in (("action_mlp_cebrier", "MLP CE+Brier", ACCENT, "o", "-"),
                                    ("edge_mlp_cebrier", "edge-ID CE+Brier", RED, "s", "-.")):
            cv = e2[key]["curve_mean"]
            covs = [p["coverage"] for p in cv]
            risks = [100 * p["risk"] for p in cv]
            # grayscale/CVD-safe: the two arms were ~17% luminance apart on hue alone (P5-3)
            a1.plot(covs, risks, color=c, lw=1.4, label=lbl,
                    linestyle=ls, marker=mk, markevery=15, ms=3, mfc="white", mec=c)
    pr = jload("e2_layaprobe_results.json")
    if pr:
        pv = pr.get("action_layaprobe_cebrier", {})
        if pv.get("curve_mean"):
            cv = pv["curve_mean"]
            a1.plot([p["coverage"] for p in cv], [100 * p["risk"] for p in cv], color=PURPLE,
                    lw=1.4, label="probe CE+Brier")
        elif pv.get("risk_at_cov_mean"):
            # probe JSON keeps coarse risk@coverage points, not a dense curve
            rc = pv["risk_at_cov_mean"]
            ks = sorted(rc, key=float)
            a1.plot([float(k) for k in ks], [100 * rc[k] for k in ks], color=PURPLE,
                    lw=1.1, ls="--", marker="o", ms=2, label="probe CE+Brier")
    a1.set_xlabel("coverage")
    a1.set_ylabel("selective risk (%)")
    a1.set_title("(b) risk\u2013coverage", fontsize=9)
    if len(a1.lines):
        h, lbl = a1.get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.28)
        fig.legend(h, lbl, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.03),
                   fontsize=6, handlelength=1.2, labelspacing=0.2,
                   columnspacing=0.8)
    save(fig, "fig_router")


def pct1_from_rate(rate, n):
    """1dp percent from a 4dp-stored rate, recovering the integer count first
    (the sole integer consistent with the 4dp value when it is precise enough),
    so stored rounding onto a .x5 tie cannot flip the displayed decimal."""
    c = round(rate * n)
    return f"{100 * c / n:.1f}"


def gap_pair_c1c3(g1, n):
    """3dp count-derived displays of the cost-1 and cost-3 gap cells, derived
    jointly from a single recovered integer wrong-hop count (c3 = 3 * c1 is a
    settled identity -- see the gap identity in App.~B), avoiding a second
    independent rounding of the 4dp-stored c3 value."""
    c1 = round(g1 * n)
    return f"{c1 / n:.3f}", f"{3 * c1 / n:.3f}"


def pct1_from_perseed(rates, denom):
    """1dp percent pooling integer counts recovered per seed (each rate is a
    4dp-stored per-seed rate over `denom` rows), not just formatting the mean."""
    total = sum(round(r * denom) for r in rates)
    return f"{100 * total / (denom * len(rates)):.1f}"


def pct1_from_counts(hits, n):
    """1dp percent from known exact integer counts (no recovery needed)."""
    return f"{100 * hits / n:.1f}"


# ---------------------------------------------------------------- fig 3 + tab endtoend
def tab_endtoend():
    e3 = jload("e3_replay.json")
    if not e3:
        return
    order = ["mimo26_everyhop", "typed_only", "typed+esc@tau0.55", "typed+esc@tau0.75", "typed+esc@tau0.9"]
    lines = [r"\begin{tabular}{lrrrrr}\toprule",
             r"policy & match (\%) & survival (\%) & gap $c_1$ & gap $c_3$ & LLM calls/dec.\ (\%) \\\midrule"]
    for k in order:
        r = e3.get(k)
        if not r:
            continue
        esc = f"{100 * r.get('llm_calls_per_decision', 0):.1f}"
        gc1, gc3 = gap_pair_c1c3(r['gap_per_episode']['wrong_cost_1'], 345)
        lines.append(f"{k.replace('mimo26_everyhop', 'MIMO Pro V2.6 every hop').replace('typed_only', 'hidden-state probe, CE+Brier').replace('typed+esc@tau0.55', 'typed+esc @ $\\tau{=}0.55$').replace('typed+esc@tau0.75', 'typed+esc @ $\\tau{=}0.75$').replace('typed+esc@tau0.9', 'typed+esc @ $\\tau{=}0.9$')} & "
                     f"{pct1_from_rate(r['decision_match'], 3875)} & {100 * r['episode_survival']:.1f} & "
                     f"{gc1} & {gc3} & {esc} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    save_tab(lines, "tab_endtoend")


def fig_endtoend():
    e3 = jload("e3_replay.json")
    if not e3:
        return
    keys = ["mimo26_everyhop", "typed_only", "typed+esc@tau0.55", "typed+esc@tau0.75", "typed+esc@tau0.9"]
    labs = ["MIMO Pro V2.6", "hidden-state probe", r"$\tau{=}.55$", r"$\tau{=}.75$", r"$\tau{=}.90$"]
    fig, ax = plt.subplots(figsize=(4.15, 2.2), nrows=1, ncols=3, gridspec_kw={"wspace": 0.62})
    fig.subplots_adjust(bottom=0.26)
    cols = [LADDER[0], PURPLE, PURPLE, PURPLE, PURPLE]
    for a, f, ttl in ((0, "decision_match", "(a) match (%)"),
                      (1, "episode_survival", "(b) survival (%)")):
        vals = [100 * e3[k][f] for k in keys if k in e3]
        ax[a].bar(range(len(vals)), vals, color=cols[: len(vals)], edgecolor="black", lw=0.4)
        for xi, v in enumerate(vals):  # measured zeros must read as zeros, not missing bars
            if v == 0:
                ax[a].text(xi, 0.03 * ax[a].get_ylim()[1], "0.0", ha="center", fontsize=6)  # 0.03 axis-fraction: zeros stay at their bars on both panel scales
        ax[a].set_xticks(range(len(vals)))
        ax[a].set_xticklabels(labs[: len(vals)], fontsize=6, rotation=35, ha="right")
        for i, t in enumerate(ax[a].get_xticklabels()):  # two-row stagger: odd ticks drop past one glyph band so adjacent rotated bands stay clear
            if i % 2:
                t.set_y(t.get_position()[1] - 0.18)
        ax[a].set_title(ttl, fontsize=8)
    lat = e3.get("mimo26_everyhop", {})
    gl = jload("gpu_latency.json")
    typed_ms = gl["typed_choice_predict_ms"]["p50"] if gl else 19.5  # typed-choice decode, GPU p50 (App. B)
    ax[2].bar([0, 1], [1000 * lat.get("lat_p50_s", 0.5), typed_ms], color=[LADDER[0], PURPLE],
              edgecolor="black", lw=0.4)
    ax[2].set_yscale("log")
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["MIMO Pro V2.6", "typed decode"], fontsize=6, rotation=35, ha="right")
    ax[2].set_ylabel("latency ms (log)", labelpad=1)
    ax[2].set_title("(c) latency p50", fontsize=8)
    save(fig, "fig_endtoend")


# ---------------------------------------------------------------- tab ablation
def tab_ablation():
    e2 = jload("e2_results.json")
    pr = jload("e2_layaprobe_results.json")
    e3 = jload("e3_replay.json")
    cen = jload("corpus_census.json")
    if not (e2 and pr and e3 and cen):
        return

    def cell(v):
        v_ps = v['per_seed']
        masked = pct1_from_perseed([s2['masked_acc'] for s2 in v_ps], 3875)
        return f"{masked}\\,/{v['ece15']['mean']:.3f}"

    keys = ["typed+esc@tau0.55", "typed+esc@tau0.75", "typed+esc@tau0.9"]
    calls = "/".join(f"{100 * e3[k]['llm_calls_per_decision']:.1f}" for k in keys)
    mtch = "/".join(pct1_from_rate(e3[k]['decision_match'], 3875) for k in keys)
    loop_n = cen["loop_events"]["retry"] + cen["loop_events"]["back"]
    dec_n = cen["advance_hops"] + cen["loop_events"]["back"]
    retry_n = cen["loop_events"]["retry"]
    lines = [
        r"\begin{tabular}{ll>{\raggedright\arraybackslash}p{0.55\linewidth}}\toprule",
        r" & item & measured effect (acc.\ /\ ECE-15 where two-sided) \\\midrule",
        f"A1 & loop folding & {cen['n_tool_calls']:,} positions $\\to$ {dec_n:,} decision "
        f"states ({cen['advance_hops']:,} advances + {cen['loop_events']['back']:,} kept returns); "
        f"{retry_n:,} retries dropped, {loop_n:,} loop events tagged "
        f"({100 * loop_n / cen['n_tool_calls']:.1f}\\% of steps) \\\\",
        f"A2 & feature MLP & {cell(e2['action_mlp_ce'])} $\\to$ {cell(e2['action_mlp_cebrier'])} \\\\",
        f" & edge-ID readout & {cell(e2['edge_mlp_ce'])} $\\to$ {cell(e2['edge_mlp_cebrier'])} \\\\",
        f" & hidden-state probe & {cell(pr['action_layaprobe_ce'])} $\\to$ "
        f"{cell(pr['action_layaprobe_cebrier'])} \\\\",
        f"A3 & escalation sweep & $\\tau{{=}}0.55/0.75/0.9$ $\\to$ {calls}\\% LLM calls, match {mtch}\\% \\\\",
        r"\bottomrule\end{tabular}",
    ]
    save_tab(lines, "tab_ablation")


# ---------------------------------------------------------------- fig 4 RSI
def fig_rsi():
    d = jload("e4_rsi.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot([r["round"] for r in d], [100 * r["test_acc"] for r in d],
            color=PURPLE, lw=1.6, marker="o", ms=3)
    ax.set_ylim(28.6, 33.4)
    # one annotation per (round, acc): an accepted candidate shares its update's point
    abbr = {"base fit": "base", "ungated update": "ung", "gated accepted": "acc",
            "ungated update +12% corrupt": "12% corr"}
    by_round = {}
    for r in d:
        by_round.setdefault(r["round"], []).append(r)
    for rd, rows in sorted(by_round.items()):
        y = 100 * rows[0]["test_acc"]
        lab = "+".join(dict.fromkeys(abbr.get(r["event"], r["event"]) for r in rows)) + f" (r{rd})"
        hot = any(("ROLLBACK" in r["event"] or "corrupt" in r["event"]) for r in rows)
        ax.annotate(lab, (rd, y), fontsize=7, color=(RED if hot else LADDER[3]),
                    textcoords="offset points",
                    xytext={0: (3, 13), 1: (-4, -18), 2: (3, -7), 3: (-4, 7), 4: (-4, 7)}.get(rd, (3, 7)),
                    ha="right" if rd in (1, 3, 4) else "left",
                    bbox=dict(fc="white", ec="none", pad=0.4))
    ax.set_xlabel("round")
    ax.set_ylabel("test acc. (%)")
    ax.set_title("RSI updates under 12% label corruption\n(round 3)", fontsize=9)
    save(fig, "fig_rsi")


if __name__ == "__main__":
    fig_convergence()
    tab_router()
    fig_router()
    tab_endtoend()
    fig_endtoend()
    tab_ablation()
    fig_rsi()