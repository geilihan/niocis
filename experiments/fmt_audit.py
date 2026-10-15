# Compact formatter for the 30-case audit sample -> readable text for manual adjudication.
import json
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
sample = json.loads((OUT / "audit_sample.json").read_text(encoding="utf-8"))
for k, c in enumerate(sample, 1):
    print(f"===== case {k:02d} | trace {c['trace'][:12]} | agent {c['agent']} | steps {c['n_steps']}")
    print(f"  proxy_a(end-update_goal)={c['proxy_a_completed']}  judge_b={c['judge_b_completed']}")
    print(f"  judge_evidence: {c['judge_evidence'][:220]}")
    for line in c["log"].splitlines():
        print("   ", line[:170])
    print()