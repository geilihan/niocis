"""Synthetic prototype corpus generator (NIOCIS).

Emits JSONL episodes in the experiments' documented contract plus a flat
state table with oracle next-hop labels, ambiguity levels and detour flags.
Deterministic under --seed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import (
    EDGES, FAIL_P, GOALS, NODES, RECOVERY_EDGES, STARTS, STAGE, State,
    allowed_succ, fold_episode, near_valid_competitors, oracle_next, soft_w,
    topological_neighbors,
)

CTX_DIM = 6


def sample_ctx(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=CTX_DIM)


def sim_failure(rng, node):
    return rng.random() < FAIL_P.get(node, 0.05)


def llm_choice(state: State, rng, oracle_edge: str, greedy_trap: bool, ambig: int):
    """Logged-agent edge choice: oracle with behavioral noise."""
    p_wrong = 0.02 + 0.13 * ambig
    if greedy_trap:
        p_wrong += 0.28
    if state.retries > 0:
        p_wrong += 0.06
    allowed = [v for v, _ in allowed_succ(state.node, state.ctx)]
    if not allowed:
        return oracle_edge
    if rng.random() < p_wrong:
        traps = [v for v in topological_neighbors(state.node, state.ctx) if v in allowed and v != oracle_edge]
        pool = traps or [v for v in allowed if v != oracle_edge]
        return rng.choice(pool) if pool else oracle_edge
    return oracle_edge


def greedy_trap_active(state: State, oracle_edge: str) -> bool:
    topo = topological_neighbors(state.node, state.ctx)
    return bool(topo) and topo[0] != oracle_edge


def rollout(rng, start: str, goal: str, ambig_bias: float = 0.0):
    ctx = sample_ctx(rng)
    if ambig_bias > 0:  # push into mid-auth band => A2/A3 fork states
        ctx[3] = 0.3 + 0.4 * rng.random()
    st = State(node=start, goal=goal, ctx=ctx)
    steps = []
    max_steps = 26
    while not st.terminated and st.steps < max_steps:
        oracle_edge, w = oracle_next(st)
        if oracle_edge == "":  # goal reached or dead end
            st.success = st.node == goal
            st.terminated = True
            break
        ambig = near_valid_competitors(st)
        trap = greedy_trap_active(st, oracle_edge)
        edge = llm_choice(st, rng, oracle_edge, trap, ambig)
        # environment execution: failures create retry / rollback / poll loops
        kind = "step"
        if sim_failure(rng, st.node):
            retry_n = 1 + (rng.random() < 0.35) + (rng.random() < 0.12)
            for _ in range(int(retry_n)):
                st.cost += 0.6
                st.steps += 1
                steps.append({"node": st.node, "kind": "retry", "edge": f"{st.node}->{st.node}",
                              "oracle_edge": f"{st.node}->{oracle_edge}", "ambig": ambig,
                              "detour": int(trap), "ctx": [round(float(c), 4) for c in ctx]})
            if rng.random() < 0.14:  # rollback: jump back to an earlier-stage node
                back = [u for u in NODES if STAGE[u] == max(STAGE[st.node] - 1, 0)]
                bnode = rng.choice(back) if back else st.node
                RECOVERY_EDGES.add((st.node, bnode, 1.7))
                st.cost += 1.7
                st.steps += 1
                steps.append({"node": bnode, "kind": "back", "edge": f"{st.node}->{bnode}",
                              "oracle_edge": f"{st.node}->{oracle_edge}", "ambig": ambig,
                              "detour": int(trap), "ctx": [round(float(c), 4) for c in ctx]})
                st.node = bnode
                st.visited[bnode] = st.visited.get(bnode, 0) + 1
                continue
        if edge.split(".")[0] == "wait":  # poll loop 1-4 extra iterations
            for _ in range(int(rng.geometric(0.45)) - 1):
                st.cost += 0.8
                st.steps += 1
                steps.append({"node": st.node, "kind": "retry", "edge": f"{st.node}->{st.node}",
                              "oracle_edge": f"{st.node}->{oracle_edge}", "ambig": ambig,
                              "detour": int(trap), "ctx": [round(float(c), 4) for c in ctx]})
        if rng.random() < 0.05:  # exploration backtrack
            back = [u for u in NODES if STAGE[u] == max(STAGE[st.node] - 1, 0) and u != st.node]
            if back:
                bnode = rng.choice(back)
                RECOVERY_EDGES.add((st.node, bnode, 1.9))
                st.cost += 1.9
                st.steps += 1
                steps.append({"node": bnode, "kind": "back", "edge": f"{st.node}->{bnode}",
                              "oracle_edge": f"{st.node}->{oracle_edge}", "ambig": ambig,
                              "detour": int(trap), "ctx": [round(float(c), 4) for c in ctx]})
                st.node = bnode
                st.visited[bnode] = st.visited.get(bnode, 0) + 1
                continue
        # normal hop (may fail-path to wrong edge; cost = context-soft weight)
        w_actual = soft_w(st.node, edge, st.ctx)
        st.cost += w_actual
        st.steps += 1
        steps.append({"node": edge, "kind": kind, "edge": f"{st.node}->{edge}",
                      "oracle_edge": f"{st.node}->{oracle_edge}", "ambig": ambig,
                      "detour": int(trap), "ctx": [round(float(c), 4) for c in ctx],
                      "terminal": edge == goal})
        st.visited[edge] = st.visited.get(edge, 0) + 1
        st.node = edge
        if st.node == goal:
            st.success = True
            st.terminated = True
    if st.steps >= max_steps and not st.terminated:
        st.terminated = True
    # oracle episode cost for optimality-gap denominator
    o = State(node=start, goal=goal, ctx=ctx)
    oc = 0.0
    for _ in range(max_steps):
        nx, w = oracle_next(o)
        if nx == "":
            break
        oc += w
        o.node = nx
    return {"trace_id": f"{start}=>{goal}#{len(steps)}", "start": start, "goal": goal,
            "outcome": "success" if st.success else "fail", "cost": round(st.cost, 3),
            "oracle_cost": round(max(oc, 0.01), 3), "steps": st.steps,
            "ctx": [round(float(c), 4) for c in ctx], "step_list": steps}


def generate(n_episodes: int, seed: int, ambig_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        start = rng.choice(STARTS)
        goal = rng.choice(GOALS)
        episodes.append(rollout(rng, start, goal, ambig_bias=ambig_frac if rng.random() < 0.35 else 0.0))
    return episodes


def write_corpus(out_dir: Path, seed: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": 6000, "val": 1000, "test": 3000, "rstream": 2400}
    split_seeds = {"train": 1, "val": 2, "test": 3, "rstream": 4}  # literal: no hash() salt
    meta = {"seed": seed, "splits": splits, "n_nodes": len(NODES), "n_edge_specs": len(EDGES)}
    for split, n in splits.items():
        eps = generate(n, seed + split_seeds[split])
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps({"trace_id": ep["trace_id"], "outcome": ep["outcome"],
                                    "start": ep["start"], "goal": ep["goal"],
                                    "cost": ep["cost"], "oracle_cost": ep["oracle_cost"],
                                    "steps": ep["steps"],
                                    "step_detail": ep["step_list"]}) + "\n")
        succ = sum(1 for e in eps if e["outcome"] == "success")
        nsteps = sum(e["steps"] for e in eps)
        nretry = sum(1 for e in eps for s in e["step_list"] if s["kind"] == "retry")
        nback = sum(1 for e in eps for s in e["step_list"] if s["kind"] == "back")
        meta[split + "_stats"] = {"episodes": n, "success": succ, "rate": round(succ / n, 4),
                                  "steps": nsteps, "retry_steps": nretry, "back_steps": nback}
    with open(out_dir / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "recovery_pairs.json", "w", encoding="utf-8") as f:
        json.dump(sorted([list(t) for t in RECOVERY_EDGES]), f)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/corpus")
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    m = write_corpus(Path(args.out), args.seed)
    print(json.dumps(m, indent=2))