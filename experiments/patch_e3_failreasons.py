# SANCTIONED one-field artifact patch (R13 P2 finding), in the same spirit as
# persist_seed_acc.py: run_all.e3's no-guard row was built with a HARDCODED
# "fail_reasons": {} (construction bug), while the row's own success=0.9962
# implies 3 failures and every other row balances #fails == sum(fail_reasons).
# run_all.py was already fixed at R3 to record real reasons; this retro-fits
# the frozen artifact with the unique driver-consistent assignment {"budget": 3}
# (guard_n=0 removes the revisit-cap class; the branch-lockout check fires zero
# times on this world). Wall-clock latency fields are NEVER touched.
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
p = RES / "e3_endtoend.json"
rows = json.loads(p.read_text(encoding="utf-8"))

for r in rows:
    if r["policy"] == "typed (no guard)":
        assert r["fail_reasons"] == {}, r["fail_reasons"]  # pristine hardcoded field
        n_fail = 800 - round(r["success"] * 800)
        assert n_fail == 3, n_fail  # 0.9962 * 800 -> 797 successes
        r["fail_reasons"] = {"budget": n_fail}
        print("patched typed (no guard) fail_reasons ->", r["fail_reasons"])
    else:
        n_fail = 800 - round(r["success"] * 800)
        assert sum(r["fail_reasons"].values()) == n_fail, (r["policy"], n_fail, r["fail_reasons"])

p.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("e3_endtoend.json fail_reasons balance now holds for all rows")