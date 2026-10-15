# Independent leak audit + integrity gates over the release bundle
# (run after make_release.py). ALL checks must PASS.
# 1) row counts + label bijection match the frozen originals
# 2) field completeness: every episode row keeps trace/session/agent/labels;
#    every e2 row keeps trace/cur/target/options; outcomes keep trace + verdict
# 3) join integrity: episode ids distinct 1:1 and e2/outcome traces subset-match
# 4) vocabulary sweep: every released string value is a code, an edge id, a
#    16-hex hash, or an allowlisted literal
# 5) leak sweep: no original label/id/internal substring, no IP/URL/email tokens
# 6) RESIDUALS.txt: single-token leftovers must be allowlisted; prose = dropped
#    rationale (expected, counted)
import json
import re
from pathlib import Path

_HOST = Path("/lw2/runs")
REAL = _HOST if _HOST.exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
OUT = (_HOST.parent / "release") if _HOST.exists() else Path(Path(__file__).resolve().parents[1], "release")

# R145 fix: released dict KEYS must be codes, allowlisted literals, or schema
# field names. Any new key -- internal name or schema drift -- fails the gate.
FIELD_KEYS = {
    "trace", "session", "task", "id", "labels", "kept", "events", "n_advance", "n_back",
    "n_retry", "backedges", "latencies", "t", "cur", "options", "option_edge_ids",
    "target", "target_edge_id", "target_in_opts", "kind", "min", "p50", "p90", "max",
    "mean", "n_steps", "derived_outcome", "judge_v2", "judge_evidence", "judge_latency_s",
    "action_vocab", "agent_vocab", "stop", "n_actions_total", "n_edges_total",
    "n_observations", "n_tool_calls", "n_episodes", "n_sessions", "catalog_size",
    "agents", "steps_per_episode", "loop_events", "advance_hops", "fwd_edges",
    "back_edges", "catalog_by_steps", "action", "count",
    # nested observable-state field names (e2 row["t"]) + episode/outcome fields
    "agent", "node", "goal", "step_idx", "prefix_len", "n_distinct", "revisit_ratio",
    "visit_count", "out_deg", "out_topfreq", "out_entropy",
}
RESID_LINE = re.compile(r"^\s*\d+\s+kind=(prose|token)\s+words=\s*\d+\s+sha256=[0-9a-f]{16}$")

ALLOWED = {"detection", "response", "protection", "recognition", "dlp-dispose", "qa-agent",
           "retry", "back", "advance", "complete", "completed", "blocked", "abandoned",
           "train", "val", "test", "<stop>", "<stop_edge>", "<unk_edge>", "act_stop"}
CODE = re.compile(r"act_\d{3}")
EDGE = re.compile(r"e\d{4}")
HASH = re.compile(r"[0-9a-f]{16}")

fails = []


def chk(name, cond, detail=""):
    print(("PASS " if cond else "FAIL "), name, detail)
    if not cond:
        fails.append(name)


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


orig_eps = read_jsonl(REAL / "episodes.jsonl")
rel_eps = read_jsonl(OUT / "episodes.jsonl")
chk("episode row counts", len(orig_eps) == len(rel_eps), f"{len(orig_eps)} vs {len(rel_eps)}")

orig_labels = sorted({L for e in orig_eps for L in e["labels"]})
rel_labels = sorted({L for e in rel_eps for L in e["labels"]})
chk("label universe 154 -> 154 opaque", len(orig_labels) == 154 and len(rel_labels) == 154
    and all(CODE.fullmatch(L) for L in rel_labels))

for split, exp in (("train", 12613), ("val", 1580), ("test", 3875)):
    rows = read_jsonl(OUT / f"e2_{split}.jsonl")
    chk(f"e2_{split} rows {exp}", len(rows) == exp, str(len(rows)))
    need = {"trace", "cur", "target", "options"}
    missing = [i for i, r in enumerate(rows) if not need <= set(r) or not r["options"]]
    chk(f"e2_{split} fields complete", not missing, str(missing[:3]))

ok_fields = all({"trace", "session", "agent", "labels"} <= set(e) and e["labels"] for e in rel_eps)
chk("episode fields complete", ok_fields)

rel_out = read_jsonl(OUT / "outcome_labels.jsonl")
verdicts = {r.get("derived_outcome") for r in rel_out}
chk("outcome rows complete", len(rel_out) == len(orig_eps) and
    all("trace" in r and r.get("derived_outcome") for r in rel_out), str(len(rel_out)))
chk("outcome verdicts in {completed,blocked,abandoned}",
    verdicts <= {"completed", "blocked", "abandoned"}, str(sorted(x for x in verdicts if x)))

eps_tr = [e["trace"] for e in rel_eps]
chk("episode ids distinct/hashed", len(set(eps_tr)) == len(eps_tr) == 1615
    and all(HASH.fullmatch(t) for t in eps_tr))
e2_tr = {r["trace"] for sp in ("train", "val", "test") for r in read_jsonl(OUT / f"e2_{sp}.jsonl")}
out_tr = {r["trace"] for r in rel_out}
chk("e2 traces join episodes", e2_tr <= set(eps_tr), str(len(e2_tr - set(eps_tr))))
chk("outcome traces join episodes", out_tr == set(eps_tr), str(len(out_tr ^ set(eps_tr))))

# ---- vocabulary sweep: every released string is coded/allowed ------------
# R145 fix (M1): dict KEYS are released strings too (they carried the internal
# action/agent names); sweep them and freeze the key schema.
strings = []
keys = []

def collect(v):
    if isinstance(v, str):
        strings.append(v)
    elif isinstance(v, list):
        for x in v:
            collect(x)
    elif isinstance(v, dict):
        for k, x in v.items():
            keys.append(k)
            collect(x)

collect(rel_eps)
for sp in ("train", "val", "test"):
    collect(read_jsonl(OUT / f"e2_{sp}.jsonl"))
collect(rel_out)
collect(json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8")))
collect(json.loads((OUT / "corpus_summary.json").read_text(encoding="utf-8")))

bad_vocab = sorted({s for s in strings
                    if not (CODE.fullmatch(s) or EDGE.fullmatch(s) or HASH.fullmatch(s)
                            or s in ALLOWED)})
chk("vocabulary sweep clean", not bad_vocab, str(bad_vocab[:5]))
bad_keys = sorted({k for k in keys if not (CODE.fullmatch(k) or k in ALLOWED or k in FIELD_KEYS)})
chk("vocabulary keys coded/field-allowlisted", not bad_keys, str(bad_keys[:5]))
_avk = json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8")).get("action_vocab", {})
chk("action_vocab keys are act_NNN codes",
    bool(_avk) and all(CODE.fullmatch(k) or k == "act_stop" for k in _avk), str(sorted(_avk)[:3]))
# R145: the VALUE column must not encode name order either -- it used to be the
# alphabetical rank of the internal name, so aligning codes sorted by value to a
# sorted name list inverted the mapping. Ordinals only (also pins density).
chk("action_vocab values are ordinals",
    bool(_avk) and all(isinstance(v, int) for v in _avk.values())
    and sorted(_avk.values()) == list(range(len(_avk))),
    str(sorted(_avk.values())[:5]))

# ---- leak sweep over the raw text of every released file ----------------
# R145 fix (M2): the old glob "*.json*l*" silently skipped RESIDUALS.txt
TEXT_SUFFIX = (".json", ".jsonl", ".txt", ".md")
blob = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                  for p in sorted(OUT.iterdir()) if p.suffix in TEXT_SUFFIX)
# the URL check exempts .md: LICENSE.txt/README.md legitimately cite the CC URL
# the URL check covers DATA files only: LICENSE.txt/README.md are documentation
# and legitimately cite the Creative Commons URL
blob_data = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in sorted(OUT.iterdir())
                       if p.suffix in (".json", ".jsonl"))
leaked = [L for L in orig_labels if L in blob]
chk("no original label strings", not leaked, str(leaked[:3]))
leaked_ids = [t for e in orig_eps for t in (e["trace"], e.get("session", "")) if t and t in blob]
chk("no original trace/session ids", not leaked_ids)
chk("no IP tokens", not re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", blob))
chk("no URL tokens", not re.findall(r"https?://|www\.", blob_data))
chk("no e-mail tokens", not re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", blob))
chk("no known internal substrings",
    not re.findall(r"sec-api-service|QwenPaw|sadb|prod-mde|update_goal|Session performed", blob))

agents = {e.get("agent") for e in rel_eps}
chk("agent names mapped only", agents <= ALLOWED, str(sorted(a for a in agents if a)))
chk("no missing agent values", None not in agents)

# ---- residual report review ----------------------------------------------
res = (OUT / "RESIDUALS.txt").read_text(encoding="utf-8")
single_bad, prose = [], 0
for line in res.splitlines():
    if line.startswith("#") or not line.strip():
        continue
    if not RESID_LINE.fullmatch(line):
        single_bad.append(line[:60])
    prose += 1
print(f"residuals: {prose} reported kinds; malformed/verbatim lines: {single_bad[:3] or 'none'}")
chk("residual report is text-free (counts+hashes only)", not single_bad)

print("RELEASE AUDIT:", "CLEAN" if not fails else f"{len(fails)} FAILED")