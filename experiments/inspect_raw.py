# Probe the shape of observations_raw.json without loading it blindly.
import json
from pathlib import Path

P = Path(Path(__file__).resolve().parents[1], "数据", "agent轨迹原始记录", "observations_raw.json")

with P.open("r", encoding="utf-8") as f:
    head = f.read(4000)
print("=== FIRST 4000 CHARS ===")
print(head)

# Determine container: JSON array vs JSONL vs object
with P.open("r", encoding="utf-8") as f:
    first_nonws = ""
    while True:
        ch = f.read(1)
        if not ch:
            break
        if not ch.isspace():
            first_nonws = ch
            break
print("\n=== CONTAINER START CHAR:", repr(first_nonws))

if first_nonws == "[":
    with P.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print("JSON array, n =", len(data))
    print("elem0 type:", type(data[0]))
    if isinstance(data[0], dict):
        print("elem0 keys:", sorted(data[0].keys()))
        print("elem0 sample:", json.dumps(data[0], ensure_ascii=False)[:1500])
        # key census over all elements
        from collections import Counter
        c = Counter()
        for e in data:
            if isinstance(e, dict):
                c.update(e.keys())
        print("key census:", dict(c))
        print("n total:", len(data))
elif first_nonws == "{":
    with P.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
            print("JSON object, top keys:", sorted(obj.keys())[:40])
            for k, v in list(obj.items())[:8]:
                print(f"  {k}: type={type(v).__name__}", end="")
                if isinstance(v, list):
                    print(f" len={len(v)}", end="")
                    if v and isinstance(v[0], dict):
                        print(f" elem0 keys={sorted(v[0].keys())}", end="")
                elif isinstance(v, dict):
                    print(f" keys={sorted(v.keys())[:15]}", end="")
                print()
        except json.JSONDecodeError:
            f.seek(0)
            n = 0
            sample = None
            from collections import Counter
            c = Counter()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if n <= 2:
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            c.update(rec.keys())
                        if sample is None:
                            sample = rec
                    except Exception:
                        pass
            print("JSONL-ish (whole-file object parse failed), approx lines:", n)
            if sample:
                print("rec keys:", sorted(sample.keys()) if isinstance(sample, dict) else type(sample))
                print("rec sample:", json.dumps(sample, ensure_ascii=False)[:1500])