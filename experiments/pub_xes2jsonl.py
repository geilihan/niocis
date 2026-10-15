# XES (.xes / .xes.gz / .zip holding xes) -> cases.jsonl per corpus:
# {"trace": case_id, "labels": [activity names sorted by time], "count": n}
# Activity strings preserved as-is: these are PUBLIC logs (nothing to hide),
# and E1 measures label-sequence structure, not semantics.
import gzip
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container
_HOST = Path("/lw2/pubdata")
PUB = _HOST if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")


def xes_stream(fp):
    """Yield (case_id, [activity names in timestamp order]) per trace."""
    n_auto = 0
    for _, elem in ET.iterparse(fp, events=("end",)):
        tag = elem.tag.split("}")[-1]
        if tag != "trace":
            continue
        case_id, evs = None, []
        for child in elem:
            ctag = child.tag.split("}")[-1]
            key, val = child.get("key"), child.get("value")
            if ctag == "event":
                ev = {}
                for g in child:
                    ev[g.get("key")] = g.get("value")
                name = ev.get("concept:name") or ev.get("activity")
                if name:
                    evs.append((ev.get("time:timestamp") or "", name))
            elif key in ("concept:name", "concept:instance") and case_id is None:
                case_id = val
        evs.sort(key=lambda t: t[0])
        if evs:
            # never silently drop a populated trace: software logs label traces
            # concept:instance and some exports carry no trace-level id at all
            n_auto += 1
            yield case_id or f"trace-{n_auto}", [n for _, n in evs]
        elem.clear()


def open_maybe_gz(p):
    if p.suffix == ".gz":
        return gzip.open(p, "rb")
    return open(p, "rb")


manifest = json.loads((PUB / "fetch_manifest.json").read_text(encoding="utf-8"))
for name, info in manifest.items():
    if not info.get("file"):
        continue
    p = PUB / info["file"]
    cases = []
    try:
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                names_in = z.namelist()
                # priority: plain xes > gzipped xes > mxml; statechart ships
                # <log>.xes.gz inside the zip alongside csv dumps
                member = next((m for m in names_in if m.lower().endswith(".xes")), None) \
                    or next((m for m in names_in if m.lower().endswith(".xes.gz")), None) \
                    or next((m for m in names_in if m.lower().endswith(".mxml")), None)
                if member is None:
                    raise ValueError("no .xes/.mxml member in zip")
                raw = z.open(member)
                fp = gzip.GzipFile(fileobj=raw) if member.lower().endswith(".gz") else raw
                for cid, labels in xes_stream(fp):
                    cases.append({"trace": cid, "labels": labels, "count": len(labels)})
        else:
            with open_maybe_gz(p) as fp:
                for cid, labels in xes_stream(fp):
                    cases.append({"trace": cid, "labels": labels, "count": len(labels)})
    except Exception as e:
        # one bad corpus must never sink the rest of the conversion
        print(f"[skip] {name}: {type(e).__name__} (corrupt/partial input)")
        continue
    out = PUB / f"{name}_cases.jsonl"
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    ev = sum(c["count"] for c in cases)
    labs = len({L for c in cases for L in c["labels"]})
    print(f"{name:12s} cases {len(cases):6d}  events {ev:7d}  labels {labs:5d}")