"""Shared definitions for the NIOCIS prototype corpus and experiments.

A "world" = loop-collapsed action graph (DAG skeleton + recovery back-edges)
with contextual edge costs; an "episode" = a walk from a start node to a goal
node produced by a noisy oracle policy (the logged agent).

Context axes (all in [0,1]): 0 slack (high = not urgent), 1 payload_stability
(high = small/stable), 2 deps (high = has dependencies), 3 auth (high =
authorized), 4 complexity (high = complex), 5 noise.

Edge guards (axis, thr): the edge is *cheap and natural* when ctx[axis] >= thr
for all guards; each violated guard multiplies the traversed weight by 2.6.
Optimal next-hop therefore depends on context (e.g. low auth must detour via
approval), which is what the router has to learn from logs.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

VIOL_MULT = 2.6

TOOLS_ACTIONS = [
    (0, "parse.intake"), (0, "parse.schema_check"), (0, "parse.triage"),
    (1, "jira.search"), (1, "jira.read"), (1, "mail.read"), (1, "fs.read"),
    (1, "http.get"), (1, "db.query"), (1, "wait.poll_status"),
    (2, "db.query_deep"), (2, "http.get_nested"), (2, "calc.aggregate"),
    (2, "calc.validate"), (2, "kb.retrieve"), (2, "wait.poll_deps"),
    (3, "plan.draft"), (3, "plan.revise"), (3, "auth.request_approval"),
    (3, "plan.select_template"), (3, "shell.exec"), (3, "http.post"),
    (3, "db.write"), (3, "mail.send"), (3, "jira.update"), (3, "fs.write"),
    (4, "report.render"), (4, "report.pdf"), (4, "report.publish"),
    (4, "jira.comment"), (4, "mail.digest"), (4, "db.audit_log"),
    (4, "report.archive"), (4, "cleanup.release_lock"),
]

NODES = [n for _, n in TOOLS_ACTIONS]
NODE_INDEX = {n: i for i, n in enumerate(NODES)}
STAGE = {n: s for s, n in TOOLS_ACTIONS}

EDGES: list[tuple[str, str, float, tuple[tuple[int, float], ...]]] = []


def _e(u, v, w=1.0, guards=()):
    EDGES.append((u, v, w, tuple(guards)))


# --- stage 0 -> 1 ---
_e("parse.intake", "parse.schema_check", 1.0)
_e("parse.intake", "parse.triage", 1.2)
_e("parse.schema_check", "jira.read", 1.0, ((1, 0.45),))     # stable payload -> direct read
_e("parse.schema_check", "jira.search", 1.1)                  # detour: search first
_e("jira.search", "jira.read", 0.8)
_e("parse.triage", "jira.read", 1.5)
_e("parse.triage", "mail.read", 1.1)
_e("jira.read", "fs.read", 0.5)
# --- stage 1 -> 2 ---
_e("jira.search", "kb.retrieve", 1.2, ((4, 0.55),))           # complex -> kb
_e("mail.read", "mail.read", 1.7)                             # poll-like retry loop
_e("mail.read", "db.query", 1.4)
_e("fs.read", "calc.validate", 1.1)
_e("http.get", "calc.aggregate", 1.3, ((2, 0.5),))            # no deps -> direct
_e("http.get", "http.get_nested", 1.0)                        # detour: nest
_e("http.get_nested", "calc.aggregate", 1.0)
_e("db.query", "calc.aggregate", 1.2, ((4, 0.5),))            # simple -> direct
_e("db.query", "db.query_deep", 1.0)                          # detour: deep query
_e("db.query_deep", "calc.aggregate", 0.9)
_e("db.query_deep", "db.query_deep", 1.8)                     # slow-query retry loop
_e("wait.poll_status", "wait.poll_status", 1.6)               # poll loop
_e("wait.poll_status", "calc.validate", 1.2)
# --- stage 2 -> 3 ---
_e("calc.aggregate", "calc.validate", 0.9)
_e("calc.validate", "plan.draft", 1.0, ((0, 0.6),))           # slack -> freehand draft
_e("calc.validate", "plan.select_template", 1.1)              # detour: quick template
_e("kb.retrieve", "plan.draft", 1.2)
_e("wait.poll_deps", "wait.poll_deps", 1.5)                   # poll loop
_e("wait.poll_deps", "plan.select_template", 1.2)
# --- stage 3 fork: direct commit vs. approval detour ---
_e("plan.draft", "db.write", 1.2, ((3, 0.6),))                # authorized -> direct
_e("plan.draft", "auth.request_approval", 1.0)                # detour: approval
_e("auth.request_approval", "db.write", 0.6)
_e("auth.request_approval", "auth.request_approval", 2.0)     # approval polling loop
_e("plan.revise", "plan.revise", 1.9)                          # refinement loop
_e("plan.revise", "db.write", 1.3, ((3, 0.6),))
_e("plan.revise", "auth.request_approval", 1.4)
_e("plan.select_template", "mail.send", 1.1)
_e("plan.select_template", "shell.exec", 1.4)
_e("shell.exec", "fs.write", 1.0)
_e("shell.exec", "report.render", 1.2)
_e("http.post", "report.render", 1.0)
_e("http.post", "http.post", 2.1)                              # flaky endpoint retry loop
_e("db.write", "db.audit_log", 0.8)
_e("db.write", "report.render", 1.1)
_e("mail.send", "mail.digest", 1.0)
_e("fs.write", "report.render", 1.1)
_e("jira.update", "jira.comment", 1.0)
# --- stage 4 exits ---
_e("db.audit_log", "report.archive", 0.9)
_e("db.audit_log", "cleanup.release_lock", 1.1)
_e("report.render", "report.pdf", 0.9)
_e("report.render", "report.publish", 1.2)
_e("report.pdf", "report.archive", 1.0)
_e("mail.digest", "report.archive", 1.0)
_e("jira.comment", "report.archive", 1.2)
_e("jira.comment", "cleanup.release_lock", 1.3)
_e("report.publish", "report.archive", 1.0)
_e("report.archive", "cleanup.release_lock", 0.9)

RECOVERY_EDGES: set[tuple[str, str, float]] = set()
# terminals re-open via a recovery back-edge ("task redirect") — the dynamic
# rollback edges recorded during episodes join the same set.
RECOVERY_EDGES.add(("report.archive", "plan.draft", 1.5))
RECOVERY_EDGES.add(("cleanup.release_lock", "plan.draft", 1.5))
RECOVERY_EDGES.add(("report.publish", "plan.draft", 1.5))
RECOVERY_EDGES.add(("mail.digest", "plan.draft", 1.5))

PRED: dict[str, list[str]] = {n: [] for n in NODES}
for u, v, _, _ in EDGES:
    PRED.setdefault(v, []).append(u)
SUCC: dict[str, list[tuple[str, float, tuple]]] = {n: [] for n in NODES}
for u, v, w, g in EDGES:
    SUCC.setdefault(u, []).append((v, w, g))

GOALS = ["report.archive", "cleanup.release_lock", "report.publish", "mail.digest"]
STARTS = ["parse.intake", "parse.schema_check", "parse.triage"]

FAIL_P = {n: 0.04 for n in NODES}
for n in ["shell.exec", "http.post", "http.get", "http.get_nested", "db.write",
          "fs.write", "wait.poll_status", "wait.poll_deps"]:
    FAIL_P[n] = 0.18
FAIL_P["shell.exec"] = 0.26


@dataclass
class State:
    node: str
    goal: str
    ctx: np.ndarray
    retries: int = 0
    visited: dict = field(default_factory=dict)
    cost: float = 0.0
    steps: int = 0
    terminated: bool = False
    success: bool = False


_REC_CACHE = {"n": -1, "map": {}}


def _rec_map():
    """(source node -> sorted [(target, weight)]) for recovery edges; rebuilt on growth."""
    if _REC_CACHE["n"] != len(RECOVERY_EDGES):
        m = {}
        for (u, v, w) in RECOVERY_EDGES:
            m.setdefault(u, []).append((v, w))
        _REC_CACHE["map"] = {k: sorted(vv) for k, vv in m.items()}
        _REC_CACHE["n"] = len(RECOVERY_EDGES)
    return _REC_CACHE["map"]


def guard_viol(guards, ctx) -> int:
    return sum(1 for c, thr in guards if ctx[c] < thr)


def edge_spec(u, v):
    for a, b, w, g in EDGES:
        if a == u and b == v:
            return w, g
    return None, None


def soft_w(u, v, ctx) -> float:
    for (vv, ww) in _rec_map().get(u, ()):
        if vv == v:
            return ww  # recovery hops carry their recorded weight
    w, g = edge_spec(u, v)
    if w is None:
        return 1.5  # unrecorded dynamic recovery hop
    return w * (VIOL_MULT ** guard_viol(g, ctx))


def allowed_succ(node, ctx) -> list[tuple[str, float]]:
    out = [(v, soft_w(node, v, ctx)) for (u, v, w, g) in EDGES if u == node]
    out.extend(_rec_map().get(node, ()))  # cached, deterministically sorted
    return out


def oracle_next(state: State) -> tuple[str, float]:
    """Context-weighted shortest path from state.node to state.goal -> next hop."""
    rev: dict[str, list[tuple[str, float]]] = {n: [] for n in NODES}
    for u in NODES:
        for v, w in allowed_succ(u, state.ctx):
            rev[v].append((u, w))
    dist = {state.goal: 0.0}
    pq = [(0.0, state.goal)]
    prev: dict[str, tuple[str, float]] = {}
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, math.inf):
            continue
        for u, w in rev[node]:
            nd = d + w
            if nd < dist.get(u, math.inf):
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
                prev[u] = (node, w)
    if state.node == state.goal or state.node not in prev:
        return ("", 0.0)
    nxt, w = prev[state.node]
    return nxt, w


def topological_neighbors(node, ctx) -> list[str]:
    """Locally attractive hops (stage progress), ignoring context costs."""
    cands = [v for u, v, _, _ in EDGES if u == node and STAGE[v] > STAGE[node]]
    if not cands:
        cands = [v for u, v, _, _ in EDGES if u == node]
    return sorted(set(cands), key=lambda v: (-STAGE[v], v))  # total order: deterministic ties


def near_valid_competitors(state: State) -> int:
    """A0-A3 analogue: #non-oracle hops whose cost is within one violated fact."""
    nxt, _ = oracle_next(state)
    comp = 0
    for u, v, w, g in EDGES:
        if u != state.node or v == nxt:
            continue
        if guard_viol(g, state.ctx) <= 1:
            comp += 1
    return min(comp, 3)


def fold_episode(steps: list[dict]) -> tuple[list[str], list[str]]:
    """Loop folding over an episode's step labels -> (skeleton, loop events)."""
    skeleton, events, last = [], [], None
    for s in steps:
        label = s["node"]
        if label == last and s.get("kind") == "retry":
            events.append(f"retry:{label}")
            continue
        if label in skeleton:
            events.append(f"back:{label}")
            continue
        skeleton.append(label)
        last = label
    return skeleton, events