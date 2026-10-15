# `experiments/real/` --- intentionally not part of this repository

The paper's Reproducibility statement lists the frozen artifacts it ran against under this
path (`episodes.jsonl`, `e2_*.jsonl`, `e1_saturation.json`, the `pub_*` results, the per-stream
`*_e2_results.json` / `*_e2_meta.json` pairs, `feats_laya_cls.npz`, ...).

Those artifacts are derived from a production SOC telemetry export that cannot be
redistributed, so they are **not** published here. What *is* published is the pseudonymized
derived corpus in `release/`, which is sufficient to reproduce every corpus-level number in
the paper --- run `python experiments/e21_release_reproduction.py`.

If you are the author working locally, this directory is where those frozen artifacts live;
it is git-ignored on purpose.
