# E5 figures + table from runs/pub_e1_results.json (and later pub_e2_results.json).
# fig_pub_sat: one row per public corpus -- (a) node/pair saturation (log x),
# (b) loop rates. tab_pub: corpus census + fold totals. Dual-path like the
# main renderer (container /lw2/runs vs local paper/ + experiments/real/).
import os
import json
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HOST = Path("/lw2/runs")
_PAPER = Path(__file__).resolve().parent.parent / "paper"
REAL = _HOST if _HOST.exists() else Path(__file__).resolve().parent / "real"
FIG = (_HOST / "figures") if _HOST.exists() else _PAPER / "figures"
TAB = (_HOST / "tables") if _HOST.exists() else _PAPER / "tables"

LADDER = ["#2B2B2B", "#4D4D4D", "#6A6A6A", "#979797", "#C6C6C6", "#EFEFEF"]
ACCENT, ACCENT2, RED = "#029E73", "#2E75B6", "#DE8F05"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "cm"})

d = json.loads((REAL / "pub_e1_results.json").read_text(encoding="utf-8"))
names = sorted(d)
# public-label display names (the JSON keys are lowercase ids); the prose uses BPIC'12/'17
DISPLAY = {"bpi12": "BPIC'12", "bpi17": "BPIC'17"}


def disp(n):
    return DISPLAY.get(n, n)

# one 3-panel block with one colored line per corpus; the 2026-09-29 preview of
# the per-corpus twin-axis rows piled every right-hand label into the gutter
PANELS = [("nodes", "count", "(a)", "node discovery"),
          ("fwd", "count", "(b)", "forward pairs"),
          ("loops_per_100_steps", "loop events / 100 steps", "(c)", "loop rate")]
# Okabe-Ito hues + distinct markers so every panel survives grayscale and CVD print
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

fig, axs = plt.subplots(figsize=(5.5, 2.05), ncols=3, sharex=True)
# corpora with <20 cases (statechart is one 17.9k-event session) cannot support a
# subsampling curve -- they stay in the census table only
fig_names = [n for n in names if d[n]["totals"]["cases"] >= 20]
# zero-parameter occupancy null (exact under case-level sampling without
# replacement) overlaid dashed on panel (a) from runs/pub_coupon.json
try:
    coupon = json.loads((REAL / "pub_coupon.json").read_text(encoding="utf-8"))
except Exception:
    coupon = None
for ax, (key, ylab, tag, title) in zip(axs, PANELS):
    for ci, name in enumerate(fig_names):
        ks = d[name]["sizes"]
        ys = [d[name]["curve"][str(k)][key]["p50"] for k in ks]
        ax.plot(ks, ys, color=COLORS[ci % len(COLORS)], lw=1.2,
                marker=MARKERS[ci % len(MARKERS)], ms=3.5, mew=0.4,
                mec="black",
                label=disp(name) if key == "nodes" else None)
        if key == "nodes" and coupon and name in coupon:
            c = coupon[name]
            ax.plot(c["sizes"], c["pred_nodes"], color=COLORS[ci % len(COLORS)],
                    lw=0.95, ls=(0, (2, 1.5)), alpha=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_ylabel(ylab if key != "nodes" else ylab)
    ax.set_title(f"{tag} {title}", fontsize=8)
    # x decade exponents render at 0.7x the label size; 8.8 keeps them >=6pt at ~1x
    ax.tick_params(axis="x", labelsize=8.8)
axs[1].set_xlabel("cases (log)")
h, l = axs[0].get_legend_handles_labels()
if coupon:
    # in-panel note instead of a gray legend proxy: every null curve is drawn in
    # its corpus color, so a gray proxy would promise a series that does not exist
    axs[0].text(0.96, 0.5, "dashed = occupancy null", transform=axs[0].transAxes,
                fontsize=6.5, ha="right", va="center",
                bbox=dict(fc="white", ec="none", alpha=1.0, pad=0.6))
fig.legend(h, l, loc="lower center", ncol=min(len(fig_names), 4), fontsize=6, frameon=False,
           bbox_to_anchor=(0.5, -0.03), handlelength=1.6, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(FIG / "fig_pub_sat.pdf")
fig.savefig(FIG / "fig_pub_sat.png", dpi=150)
plt.close(fig)
print("wrote fig_pub_sat")

lines = [r"\begin{tabular}{lrrrrrrr}\toprule",
         r"corpus & cases & events & labels & advance & retry & back & loop mass (\%) \\\midrule"]
for name in names:
    t = d[name]["totals"]
    lines.append(f"{disp(name)} & {t['cases']:,} & {t['events']:,} & {t['labels']:,} & "
                 f"{t['advance']:,} & {t['retry']:,} & {t['back']:,} & "
                 f"{100 * t['loop_mass']:.1f} \\\\")
lines.append(r"\bottomrule\end{tabular}")
(TAB / "tab_pub.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote tab_pub")

# ---- optional: catalog-size sweep (item #4) --------------------------------
try:
    sw = json.loads((REAL / "pub_sweep_results.json").read_text(encoding="utf-8"))
except Exception:
    sw = None
if sw:
    # the verdict view: class-minus-edge margin vs catalog size. Toolbench-train
    # provides the only multi-budget axis; the 16-entry class-vs-edge legend of
    # the first draft ate half the panel and hid every curve.
    fig2, ax2 = plt.subplots(figsize=(4.6, 2.15))
    from collections import defaultdict
    series = {}
    for corpus, rows in sw.items():
        by_c = defaultdict(dict)
        for r in rows:
            by_c[r["catalog"]][r["space"]] = r
        pts = [(c, 100 * (by_c[c]["action"]["masked_acc"]["mean"]
                          - by_c[c]["edge"]["masked_acc"]["mean"]))
               for c in sorted(by_c) if "action" in by_c[c] and "edge" in by_c[c]]
        if pts:
            series[corpus] = pts
    multi = {k: v for k, v in series.items() if len(v) > 1}
    single = {k: v for k, v in series.items() if len(v) == 1}
    ax2.axhline(0, color=LADDER[2], lw=0.9, ls="--", zorder=1)
    for ci, (corpus, pts) in enumerate(sorted(multi.items())):
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, color=ACCENT2, lw=1.5, marker="o", ms=4, zorder=3,
                 label=f"{disp(corpus)} (sweep axis)")
    x1, y1 = zip(*[(x, y) for _, v in sorted(single.items()) for x, y in v
                   if 10 <= x <= 40])
    ax2.plot(x1, y1, "s", color=ACCENT, ms=4, zorder=3,
             label="process logs (training vocabularies)")
    x2, y2 = zip(*[(x, y) for _, v in sorted(single.items()) for x, y in v if x > 40])
    ax2.plot(x2, y2, "D", color=RED, ms=4.5, zorder=3,
             label="toolbench_eval (single budget)")
    ax2.set_xscale("log")
    ax2.set_xlabel("catalog size C (log)")
    ax2.set_ylabel("class − edge-ID acc. (pts)")
    ax2.set_ylim(-3.6, 2.4)
    ax2.set_title("class margin over edge-ID vs catalog size", fontsize=8)
    ax2.legend(framealpha=0.9, labelspacing=0.25, borderpad=0.3, fontsize=7,
               loc="lower right")
    fig2.tight_layout()
    fig2.savefig(FIG / "fig_pub_sweep.pdf")
    fig2.savefig(FIG / "fig_pub_sweep.png", dpi=150)
    plt.close(fig2)
    print("wrote fig_pub_sweep")