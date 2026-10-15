# Schema-tolerant parser: tool-use trajectories -> cases.jsonl per dataset
# ({trace, labels:[api names in call order]}). Two input shapes:
#   <ds>.json / <ds>.jsonl / <ds>.parquet   single big files (chain extraction)
#   <ds>_raw/*.jsonl                        one dialogue session per file
#                                           (API-Bank: turns with role=="API")
# Fingerprint of matched key paths stored in tool_parse_report.json.
import gzip
import json
import re
from pathlib import Path

# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container
_host = Path("/lw2/pubdata")
PUB = _host if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")

NAME_KEYS = ("name", "api_name", "tool", "tool_name", "api", "function_name")
CHAIN_KEYS = ("api_calls", "calls", "tool_calls", "steps", "actions", "plan",
              "trajectory", "function_calls", "answers")


def step_name(st):
    if not isinstance(st, dict):
        return None
    for k in NAME_KEYS:
        v = st.get(k)
        if isinstance(v, str):
            return v
    fn = st.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return None


def chains_from(rec):
    out, paths = [], []
    if not isinstance(rec, dict):
        return out, paths
    for key in CHAIN_KEYS:
        v = rec.get(key)
        if isinstance(v, dict):
            v = [v]
        if isinstance(v, list) and v:
            if all(isinstance(s, dict) for s in v):
                ch = [n for n in (step_name(s) for s in v) if n]
                if ch:
                    out.append(ch); paths.append(key)
            elif all(isinstance(s, list) for s in v):
                for sub in v:
                    ch = [n for n in (step_name(s) for s in sub if isinstance(s, dict)) if n]
                    if ch:
                        out.append(ch); paths.append(key + "[]")
        if out:
            return out, paths
    # dialogue form: tool/function/API turns in order (API-Bank style),
    # plus ToolBench's ReAct text blocks ("Action: <api_name>" lines in value)
    for key in ("messages", "conversation", "conversations", "dialogue", "text"):
        v = rec.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            ch = []
            hit_paths = set()
            for turn in v:
                role = turn.get("role") or turn.get("from")
                if str(role).lower() in ("tool", "function", "assistant", "api"):
                    n = step_name(turn) or step_name(turn.get("function_call") or {})
                    if n:
                        ch.append(n)
                        hit_paths.add(key + "/tool-turns")
                if str(role).lower() == "assistant":
                    text = turn.get("value") if isinstance(turn.get("value"), str) else \
                        (turn.get("text") if isinstance(turn.get("text"), str) else "")
                    # (?! Input) guards "Action Input: {...}" false matches
                    hits = [m.strip() for m in re.findall(r"Action:(?! Input)[ \t]*([^\n]+)", text)]
                    hits = [h for h in hits if h and " " not in h]
                    if hits:
                        ch.extend(hits)
                        hit_paths.add(key + "/ReAct-Action")
            if ch:
                return [ch], sorted(hit_paths)
    return out, paths


report = {}

# ---- multi-file dirs: one .jsonl dialogue session per file -------------------
for sub in sorted(PUB.glob("*_raw")):
    files = sorted(sub.glob("*.jsonl"))
    cases = []
    for f in files:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        ch, paths = [], set()
        for rec in rows:
            if isinstance(rec, dict) and str(rec.get("role", "")).lower() == "api":
                n = step_name(rec)
                if n:
                    ch.append(n); paths.add("role=API/" + n)
        # fall back to generic dialogue extraction if no API turns matched
        if not ch:
            for cand in chains_from({"dialogue": rows})[0]:
                ch = cand; break
            if ch:
                paths.add("dialogue/generic")
        if ch:
            cases.append({"trace": f"{sub.name}-{f.stem}", "labels": ch, "count": len(ch)})
    name = sub.name[:-4] if sub.name.endswith("_raw") else sub.name
    if cases:
        (PUB / f"{name}_cases.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    labs = {L for c in cases for L in c["labels"]}
    report[f"{sub.name}/"] = {"records": len(files), "cases": len(cases), "labels": len(labs),
                              "key_paths": ["role=API/api_name", "dialogue/generic"]}
    print(f"{sub.name + '/':28s} records {len(files):7d} -> cases {len(cases):7d} "
          f"labels {len(labs):6d}")

# ---- single big files: chain extraction per record ---------------------------
for p in sorted(PUB.glob("*")):
    if p.is_dir():
        continue
    if p.name.endswith(("_cases.jsonl", ".json")) and ("manifest" in p.name or "report" in p.name):
        continue
    if p.suffix not in (".json", ".jsonl", ".parquet") and not p.name.endswith(".json.gz"):
        continue
    if p.name.endswith("_cases.jsonl"):
        continue
    rows = []
    try:
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(p).to_pylist()
        elif p.name.endswith(".gz"):
            rows = [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]
        elif p.suffix == ".jsonl":
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else next(
                (v for v in obj.values() if isinstance(v, list)), [])
    except Exception as e:
        report[p.name] = {"error": type(e).__name__}
        print(f"[skip] {p.name}: {type(e).__name__}")
        continue

    cases, paths = [], []
    for i, rec in enumerate(rows):
        for j, ch in enumerate(chains_from(rec)[0]):
            cases.append({"trace": f"{p.stem}-{i}-{j}", "labels": ch, "count": len(ch)})
        for q in chains_from(rec)[1]:
            paths.append(q)
    name = p.name.split(".")[0]
    if cases:
        (PUB / f"{name}_cases.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    labs = {L for c in cases for L in c["labels"]}
    report[p.name] = {"records": len(rows), "cases": len(cases), "labels": len(labs),
                      "key_paths": sorted(set(paths))}
    print(f"{p.name:28s} records {len(rows):7d} -> cases {len(cases):7d} labels {len(labs):6d} "
          f"paths {sorted(set(paths))[:3]}")

(PUB / "tool_parse_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print("wrote tool_parse_report.json")