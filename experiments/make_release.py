# Public release bundle: pseudonymize the frozen real artifacts (structure only).
# Policy:
#   id-shaped FIELDS (trace/session/task keys) -> sha256[:16] regardless of value shape
#   action labels -> act_001.. (stable sorted order; mapping WITHHELD)
#   edge ids e####, <stop_edge>, <unk_edge> -> kept (already opaque structural codes)
#   tool-argument payloads / free-text rationale -> dropped wholesale (RESIDUALS.txt)
# Gates in verify_release.py. Runs in the remote container (/lw2/runs ->
# /lw2/release) or locally (experiments/real -> ../release).
import hashlib
import json
import re
from pathlib import Path

_HOST = Path("/lw2/runs")
REAL = _HOST if _HOST.exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
OUT = (_HOST.parent / "release") if _HOST.exists() else Path(Path(__file__).resolve().parents[1], "release")
OUT.mkdir(parents=True, exist_ok=True)

ID_KEYS = {"trace", "session", "trace_id", "session_id", "task", "task_id", "id"}


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def hid(s):
    return hashlib.sha256((SALT + "|" + s).encode()).hexdigest()[:16]


# private salt: withheld from the release alongside the name mapping, so
# low-entropy internal ids cannot be dictionary-recovered from the hashes
_salt_path = REAL / "release_salt.txt"
if _salt_path.exists():
    SALT = _salt_path.read_text(encoding="utf-8").strip()
else:
    import os
    SALT = os.urandom(32).hex()
    _salt_path.write_text(SALT + "\n", encoding="utf-8", newline="\n")


eps = read_jsonl(REAL / "episodes.jsonl")
labels = sorted({L for e in eps for L in e["labels"]})
STOP = "<stop>"
# Code assignment must NOT be a function of label ORDER: an alphabetical
# enumeration (act_{i+1} = sorted(labels)[i]) is invertible by anyone who knows
# the label set, which would make the withheld name mapping recoverable without
# the salt. Codes are therefore a salt-derived permutation, and the assignment
# is asserted not to coincide with the alphabetical enumeration.
_rank = sorted(range(len(labels)),
               key=lambda i: hashlib.sha256(f"{SALT}|code|{labels[i]}".encode()).hexdigest())
assert _rank != list(range(len(labels))), "code assignment degenerated to alphabetical order"
AMAP = {labels[i]: f"act_{r + 1:03d}" for r, i in enumerate(_rank)}
AMAP[STOP] = "act_stop"
AGENTS = {"QwenPaw_QA_Agent_0.2": "qa-agent"}
EDGE_OK = re.compile(r"e\d{4}")
ALLOWED_RESIDUALS = {
    "detection", "response", "protection", "recognition", "dlp-dispose", "qa-agent",
    "retry", "back", "advance", "complete", "completed", "blocked", "abandoned",
    "train", "val", "test", "<stop>", "<stop_edge>", "<unk_edge>", "act_stop",
}

residuals = {}


def note(v):
    residuals[v] = residuals.get(v, 0) + 1


def tx(v, key=None):
    """Recursive pseudonymize; unknown strings are dropped and logged."""
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        if key in ID_KEYS:
            return hid(v)
        if v in AMAP:
            return AMAP[v]
        if v in AGENTS:
            return AGENTS[v]
        if EDGE_OK.fullmatch(v) or v in ("<stop_edge>", "<unk_edge>"):
            return v
        if re.fullmatch(r"[0-9a-f]{16,}", v):
            return hid(v)
        if v in ALLOWED_RESIDUALS:
            return v
        note(v)
        return None
    if isinstance(v, list):
        return [w for w in (tx(x) for x in v) if w is not None]
    if isinstance(v, dict):
        return {k: w for k, w in ((k, tx(x, k)) for k, x in v.items()) if w is not None}
    note(type(v).__name__)
    return None


def write_jsonl(p, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8", newline="\n")


# ---- episodes: explicit field allowlist (t0/inputs dropped: arguments may ----
# carry IPs, hostnames, alert ids; latencies kept as plain numbers)            # ----
EP_KEYS = ["trace", "session", "agent", "labels", "kept", "events", "n_advance",
           "n_back", "n_retry", "backedges", "latencies"]
out_eps = [{k: tx(e[k], k) for k in EP_KEYS if k in e} for e in eps]
write_jsonl(OUT / "episodes.jsonl", out_eps)

# ---- decision rows: recursive transform over all fields ---------------------
for split in ("train", "val", "test"):
    rows = read_jsonl(REAL / f"e2_{split}.jsonl")
    write_jsonl(OUT / f"e2_{split}.jsonl", [tx(r) for r in rows])

# ---- meta: CODED vocabularies only (the edge vocab is a pure function of the fold)
# R145 fix (M1): tx() never transformed dict KEYS, so the released vocabularies
# carried internal action/agent names. Emit codes only; refuse to guess.
meta = json.loads((REAL / "e2_meta.json").read_text(encoding="utf-8"))
_av = meta.get("action_vocab", {})
_gv = meta.get("agent_vocab", {})
assert all(n in AMAP for n in _av), "unmapped action label in vocab"
assert all(n in AGENTS or n in ALLOWED_RESIDUALS for n in _gv), "unmapped agent name in vocab"
meta_out = {
    # values must NOT encode name order: the internal index happened to equal the
    # alphabetical rank of the name, so sorting codes by value and aligning to a
    # sorted name list inverted the mapping (154/154). Emit ordinals instead.
    "action_vocab": {AMAP[n]: int(AMAP[n][4:]) - 1 for n in _av},
    "agent_vocab": {m: i for i, m in enumerate(sorted({AGENTS.get(n, n) for n in _gv}))},
    "stop": meta.get("stop", STOP),
    "n_actions_total": meta.get("n_actions_total"),
    "n_edges_total": meta.get("n_edges_total"),
}
(OUT / "e2_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=1),
                                 encoding="utf-8", newline="\n")

# ---- outcome labels (free-text rationale drops via tx) ----------------------
outs = read_jsonl(REAL / "outcome_labels_v2.jsonl")
write_jsonl(OUT / "outcome_labels.jsonl", [{k: tx(x, k) for k, x in r.items()} for r in outs])

# ---- slim corpus summary ----------------------------------------------------
census = json.loads((REAL / "corpus_census.json").read_text(encoding="utf-8"))
summary = {
    "n_observations": census["n_obs"],
    "n_tool_calls": census["n_tool_calls"],
    "n_episodes": census["n_episodes"],
    "n_sessions": census["n_sessions"],
    "catalog_size": census["catalog_size"],
    "agents": {AGENTS.get(k, k): v for k, v in census["agents"].items()
               if AGENTS.get(k, k) in ALLOWED_RESIDUALS},
    "steps_per_episode": census["steps_per_episode"],
    "loop_events": census["loop_events"],
    "advance_hops": census["advance_hops"],
    "fwd_edges": census["fwd_edges"],
    "back_edges": census["back_edges"],
    "catalog_by_steps": [{"action": AMAP[x[0]], "count": x[1]}
                         for x in census.get("catalog_by_steps", [])],
}
(OUT / "corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                        encoding="utf-8", newline="\n")

# R145 fix (M2): counts + kind + word count + salted hash; never the text.
(OUT / "RESIDUALS.txt").write_text(
    "# dropped upstream: counts, sizes and salted hashes only (no text)\n"
    + "\n".join(f"{c:6d}  kind={'prose' if len(s.split()) >= 4 else 'token'}  "
                f"words={len(s.split()):3d}  sha256={hid(s)}"
                for s, c in sorted(residuals.items(), key=lambda kv: -kv[1])) + "\n",
    encoding="utf-8", newline="\n")

print(f"episodes: {len(out_eps)}  labels codified: {len(labels)}  residual kinds: {len(residuals)}")
print(f"release written to {OUT}")