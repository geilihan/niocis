"""
收敛曲线实验脚手架（研究问题假设 1 的验证，对应立项书 E1）。

产出两条饱和曲线 + 一张环类型分布图：
  (a) loop-collapsed 图的节点数/边数 vs 轨迹量
  (b) back-edge 数量/循环率 vs 轨迹量
辅证：Inductive Miner 归纳的 Petri 网 place/transition 计数 vs 轨迹量（对照表示）

依赖: pip install pm4py networkx matplotlib numpy
用法: python experiments/convergence_scaffold.py --logs agent_logs.jsonl --out results/
数据: JSONL，每行一条轨迹：
  {"trace_id": "...", "outcome": "success|fail",
   "steps": [{"tool": "search", "action": "query", "args_sig": "h3k2",
              "status": "ok|fail|retry", "t_ms": 1200}, ...]}
  args_sig = 参数指纹（同 tool#action 但参数差异大的重试折叠成宏节点时保留区分度）
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]  # 按实际数据量裁剪
N_REPS = 20  # bootstrap 次数


def load_traces(path):
    traces = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            traces.append(rec)
    return traces


def abstract_step(step):
    """把一步归一化为语义动作标签（节点身份）。"""
    return f"{step['tool']}#{step['action']}"


def fold_trace(steps):
    """
    环折叠：把一条轨迹压成「骨架序列 + 环事件」。
    规则（可迭代）：
      - 连续同标签重试(status=retry) → 记为自环事件 retry:<label>
      - 回到更早已访问标签(间隔>1) → 记为回边事件 back:<label>，并从骨架中裁掉重复段
    返回 (skeleton_labels, events)
    """
    skeleton, events = [], []
    last_label = None
    for i, s in enumerate(steps):
        label = abstract_step(s)
        if label == last_label and s.get("status") == "retry":
            events.append(f"retry:{label}")
            continue
        if label in skeleton[:-1]:  # 非相邻的重复访问 → 回边
            events.append(f"back:{label}")
            continue
        skeleton.append(label)
        last_label = label
    return skeleton, events


def build_graph(traces):
    """折叠后骨架上的带权 DiGraph；回边单独计数。"""
    g = nx.DiGraph()
    back_counter, retry_counter = Counter(), Counter()
    for tr in traces:
        skeleton, events = fold_trace(tr["steps"])
        for a, b in zip(skeleton, skeleton[1:]):
            if g.has_edge(a, b):
                g[a][b]["weight"] += 1
            else:
                g.add_edge(a, b, weight=1)
        for n in set(skeleton):
            g.add_node(n)
        for ev in events:
            kind, label = ev.split(":", 1)
            (retry_counter if kind == "retry" else back_counter)[label] += 1
    return g, back_counter, retry_counter


def loop_stats(traces):
    """轨迹级循环率：重复访问步占比；环类型分布。"""
    rates, kinds = [], Counter()
    for tr in traces:
        labels = [abstract_step(s) for s in tr["steps"]]
        _, events = fold_trace(tr["steps"])
        dup = len(events)
        rates.append(dup / max(len(labels), 1))
        for ev in events:
            kinds[ev.split(":", 1)[0]] += 1  # retry / back
    return float(np.mean(rates)), kinds


def count_back_edges(g):
    """DFS 统计回边（回到 DFS 祖先的边）≈ 恢复边数量。"""
    n = 0
    for cycle_basis_dummy in [0]:  # 占位：回边用 nx.find_cycle 不够，用 DFS 状态
        pass
    state = {}  # 0=unvisited 1=in-stack 2=done
    count = 0

    def dfs(u):
        nonlocal count
        state[u] = 1
        for v in g.successors(u):
            if state.get(v, 0) == 1:
                count += 1
            elif state.get(v, 0) == 0:
                dfs(v)
        state[u] = 2

    for node in g.nodes:
        if state.get(node, 0) == 0:
            dfs(node)
    return count


def pm4py_induce_counts(trace_list):
    """对照表示：Inductive Miner 归纳 Petri 网，返回 (places, transitions)。失败返回 None。"""
    try:
        import pm4py
        from pm4py.objects.conversion.log import converter as log_converter
        event_log = [
            [{"concept:name": abstract_step(s)} for s in tr["steps"]]
            for tr in trace_list
        ]
        log = log_converter.apply(
            event_log, variant=log_converter.Variants.TO_EVENT_LOG
        )
        net, _, _ = pm4py.discover_petri_net_inductive(log)  # pm4py>=2.3
        return len(net.places), len(net.transitions)
    except Exception as e:  # TODO: 打通后移除宽异常
        print(f"[warn] pm4py 归纳失败: {e}")
        return None


def run_curve(traces, out_dir: Path):
    sizes = [s for s in SAMPLE_SIZES if s <= len(traces)]
    if not sizes:
        sizes = [len(traces)]
    rows = []
    for n in sizes:
        acc = {k: [] for k in ("nodes", "edges", "back_edges", "loop_rate")}
        for _ in range(N_REPS):
            sub = [traces[i] for i in np.random.choice(len(traces), n, replace=False)]
            g, _, _ = build_graph(sub)
            rate, _ = loop_stats(sub)
            acc["nodes"].append(g.number_of_nodes())
            acc["edges"].append(g.number_of_edges())
            acc["back_edges"].append(count_back_edges(g))
            acc["loop_rate"].append(rate)
        row = {"n": n}
        for k, vals in acc.items():
            row[k] = float(np.mean(vals))
            row[k + "_std"] = float(np.std(vals))
        pm = pm4py_induce_counts(sub)
        row["pn_places"], row["pn_transitions"] = pm if pm else (None, None)
        rows.append(row)
        print(row)

    with open(out_dir / "convergence.csv", "w", encoding="utf-8") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in rows[0].keys()) + "\n")

    # 画图：(a) 规模饱和 (b) 环饱和
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in [("nodes", "nodes"), ("edges", "edges"), ("back_edges", "back-edges")]:
        axes[0].errorbar(ns, [r[key] for r in rows],
                         yerr=[r[key + "_std"] for r in rows], label=label, capsize=3)
    axes[0].set_xlabel("#traces"); axes[0].set_ylabel("graph size")
    axes[0].set_title("(a) loop-collapsed graph saturation"); axes[0].legend()
    axes[1].plot(ns, [r["loop_rate"] for r in rows], marker="o", label="loop rate")
    if rows[0].get("pn_places") is not None:
        axes[1].plot(ns, [r["pn_places"] for r in rows], marker="s", label="PN places (IM)")
    axes[1].set_xlabel("#traces"); axes[1].set_title("(b) loop / control saturation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_curves.png", dpi=150)

    # 环类型分布
    _, kinds = loop_stats(traces)
    plt.figure(figsize=(5, 4))
    plt.bar(kinds.keys(), kinds.values())
    plt.title("loop kinds (full corpus)")
    plt.savefig(out_dir / "loop_kinds.png", dpi=150)
    print(f"[done] 输出已写入 {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="JSONL 轨迹文件")
    ap.add_argument("--out", default="results", help="输出目录")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.logs)
    print(f"加载 {len(traces)} 条轨迹")
    run_curve(traces, out_dir)


if __name__ == "__main__":
    main()