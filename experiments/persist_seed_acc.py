"""Persist the E4 seed (round-0) router's held-out test accuracy into e4_rsi.json.

fig_rsi already plots this point (g[0]) recomputed inline; storing it makes the
rsi-r0 paper number traceable to results/e4_rsi.json like every other fill.
Deterministic: identical split seeds, feature extraction and MLP
hyper-parameters to run_all.e4() (MLPClassifier 128,64 / max_iter=400 /
n_iter_no_change=8 / random_state=0 fit on Xtr[:400], evaluated on Xte).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_all import CORPUS, build_dataset, load  # noqa: E402
from common import RECOVERY_EDGES  # noqa: E402


def main():
    np.random.seed(0)
    # world parity with the corpus: share its discovered recovery back-edges
    rp = CORPUS / "recovery_pairs.json"
    if rp.exists():
        RECOVERY_EDGES.update(tuple(t) for t in json.loads(rp.read_text(encoding="utf-8")))
    Xtr, ytr, _ = build_dataset(load("train"))
    Xte, yte, _ = build_dataset(load("test"))
    seed = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=False,
                         n_iter_no_change=8, random_state=0).fit(Xtr[:400], ytr[:400])
    acc = float((seed.predict(Xte) == yte).mean())

    p = HERE / "results" / "e4_rsi.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["seed_test_acc"] = round(acc, 4)
    data["seed_states"] = 400
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("seed_test_acc =", data["seed_test_acc"])


if __name__ == "__main__":
    main()