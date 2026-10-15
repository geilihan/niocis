# E21: release-only reproduction check (v2 -- expectations corrected).
# Question: does the PUBLIC pseudonymized bundle (release/) alone reproduce the corpus-level
# numbers the paper reports? Nothing here reads experiments/real/ (the internal frozen artifacts),
# so a PASS is evidence that a reader holding only the public release can check those numbers.
# v2 fixes four of my own check-design errors from v1:
#   (a) 1,656 is the PRE-fold trace count; episodes.jsonl holds post-fold episodes (1,615).
#   (b) the corpus "catalog_size" lives in corpus_summary.json, not e2_meta.json.
#   (c) episodes.jsonl has no n_tool_calls/n_observations fields; those totals come from
#       corpus_summary.json and are cross-checked by summing per-episode fold events.
#   (d) the paper's mean-declared-options 40.21 is the HELD-OUT (test) mean, not the pooled mean.
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "release"
OUT = Path(__file__).resolve().parent


def jl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


eps = jl(R / "episodes.jsonl")
meta = json.loads((R / "e2_meta.json").read_text(encoding="utf-8"))
S = {s: jl(R / f"e2_{s}.jsonl") for s in ("train", "val", "test")}
cs = json.loads((R / "corpus_summary.json").read_text(encoding="utf-8"))

res = {"_protocol": {
    "inputs": "release/ only (episodes.jsonl, e2_{train,val,test}.jsonl, e2_meta.json, corpus_summary.json, outcome_labels.jsonl)",
    "not_read": "experiments/real/* (internal frozen artifacts) -- deliberately untouched",
    "expected_from": "paper/main_body.tex",
}}

res["episode_keys"] = sorted(eps[0].keys())
res["e2_keys"] = sorted(S["test"][0].keys())
res["meta_keys"] = sorted(meta.keys())
res["corpus_summary_keys"] = sorted(cs.keys())

res["episodes_rows"] = len(eps)
res["distinct_trace_postfold"] = len({e["trace"] for e in eps})
res["distinct_session"] = len({e["session"] for e in eps})
res["agent_counts"] = dict(Counter(e.get("agent") for e in eps))
res["sum_retry"] = sum(e["n_retry"] for e in eps)
res["sum_back"] = sum(e["n_back"] for e in eps)
res["sum_advance"] = sum(e["n_advance"] for e in eps)
res["sum_fold_events"] = res["sum_retry"] + res["sum_back"] + res["sum_advance"]
res["outcome_rows"] = len(jl(R / "outcome_labels.jsonl"))

sel = {s: {"n": len(v),
           "mean_options": round(sum(len(r["options"]) for r in v) / len(v), 4),
           "in_space": sum(1 for r in v if r["target"] in r["options"]),
           } for s, v in S.items()}
allrows = S["train"] + S["val"] + S["test"]
sel["all"] = {"n": len(allrows),
              "mean_options": round(sum(len(r["options"]) for r in allrows) / len(allrows), 4),
              "in_space": sum(1 for r in allrows if r["target"] in r["options"])}
res["splits"] = sel
res["mean_options_by_split"] = {s: sel[s]["mean_options"] for s in sel}
res["in_space_by_split"] = {s: sel[s]["in_space"] for s in sel}
res["vocab_len"] = len(meta.get("action_vocab") or [])
res["meta_catalog_size"] = cs.get("catalog_size")
res["meta_n_actions_total"] = meta.get("n_actions_total")
res["stop_field"] = meta.get("stop")

EXP = {"episodes": 1615, "sessions": 1611, "splits": (12613, 1580, 3875, 18068),
       "agents": (1174, 245, 139, 54), "test_in_space": 3568, "catalog": 154,
       "tool_calls": 22192, "observations": 22328, "retry": 4124, "back": 2364, "advance": 15704,
       "test_mean_options": 40.21}
CHK = {
    "episodes 1,615": res["episodes_rows"] == EXP["episodes"],
    "sessions 1,611 (corpus_summary == counted)": res["distinct_session"] == cs.get("n_sessions") == EXP["sessions"],
    "splits 12,613/1,580/3,875 (sum 18,068)": (sel["train"]["n"], sel["val"]["n"], sel["test"]["n"], sel["all"]["n"]) == EXP["splits"],
    "agent slices 1,174/245/139/54": (res["agent_counts"].get("detection"), res["agent_counts"].get("response"),
                                      res["agent_counts"].get("protection"), res["agent_counts"].get("recognition")) == EXP["agents"],
    "held-out in-space 3,568/3,875 (paper 92.1%)": sel["test"]["in_space"] == EXP["test_in_space"],
    "held-out mean declared options = 40.21 (paper)": abs(sel["test"]["mean_options"] - EXP["test_mean_options"]) < 0.005,
    "catalog 154 (corpus_summary + action_vocab)": (res["meta_catalog_size"], res["vocab_len"]) == (EXP["catalog"], EXP["catalog"]),
    "meta n_actions_total 154 or 155 (the stop option may be counted separately)": res["meta_n_actions_total"] in (154, 155),
    "stop token recorded as its own field": isinstance(res["stop_field"], str) and len(res["stop_field"]) > 0,
    "tool calls 22,192 (corpus_summary)": cs.get("n_tool_calls") == EXP["tool_calls"],
    "observations 22,328 (corpus_summary)": cs.get("n_observations") == EXP["observations"],
    "fold events 4,124 retry + 2,364 back + 15,704 advance = 22,192 (summed from episode rows)":
        (res["sum_retry"], res["sum_back"], res["sum_advance"]) == (EXP["retry"], EXP["back"], EXP["advance"])
        and res["sum_fold_events"] == EXP["tool_calls"],
}
res["checks"] = {k: bool(v) for k, v in CHK.items()}
res["checks_all_pass"] = all(res["checks"].values())
res["not_reproducible_from_release_alone"] = [
    "pre-fold trace count 1,656 (the fold collapses the trace-ID grouping; episodes.jsonl is post-fold)",
    "23 action families (semantic grouping, withheld together with the label mapping)",
    "all model-level results (need the training pipeline plus the internal feature tensor)",
]
res["internal_consistency_extra"] = {
    "corpus_summary.n_episodes == rows": cs.get("n_episodes") == res["episodes_rows"],
    "corpus_summary.loop_events == summed row fold events": cs.get("loop_events") == {"retry": res["sum_retry"], "back": res["sum_back"]},
    "corpus_summary.advance_hops == summed row advances": cs.get("advance_hops") == res["sum_advance"],
    "corpus_summary.agents == counted agents": cs.get("agents") == res["agent_counts"],
}

print("episode keys  :", res["episode_keys"])
print("e2 test keys  :", res["e2_keys"])
print("meta keys     :", res["meta_keys"])
print()
for k, v in CHK.items():
    print(("PASS " if v else "FAIL "), k)
print()
for k, v in res["internal_consistency_extra"].items():
    print(("PASS " if v else "FAIL "), "[internal]", k)
print()
print("episodes", res["episodes_rows"], "post-fold traces", res["distinct_trace_postfold"],
      "sessions", res["distinct_session"], "outcome rows", res["outcome_rows"])
print("splits", {s: sel[s]["n"] for s in sel})
print("mean options by split", res["mean_options_by_split"])
print("in-space", res["in_space_by_split"])
print("fold-event sums", res["sum_retry"], res["sum_back"], res["sum_advance"], "->", res["sum_fold_events"])
print("ALL PASS:", res["checks_all_pass"])

(OUT / "e21_release_reproduction.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT / "e21_release_reproduction.json")
