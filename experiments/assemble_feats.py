# Assemble feats_laya_cls.npz from feat_parts/*.npy produced by sharded extraction passes.
import numpy as np
from pathlib import Path

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
PARTS = OUT / "feat_parts"
FEAT = OUT / "feats_laya_cls.npz"

feats = {}
for tag in ("train", "val", "test"):
    parts = sorted(PARTS.glob(f"{tag}_*.npy"))
    arrs = [np.load(p) for p in parts]
    feats[tag] = np.concatenate(arrs)
    print(tag, feats[tag].shape, f"({len(parts)} parts)")
np.savez_compressed(FEAT, **feats)
print("saved ->", FEAT)