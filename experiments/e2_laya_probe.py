# E2 "Jev真身" arm: hidden-state readout on the Laya backbone (frozen ModernBERT-large encoder).
# Designated readout position = h[:, 0] (the position Laya's own pooled act head reads);
# linear head over the declared label logits (action classes 155 / structural-edge classes 2,140),
# trained with CE vs CE+Brier proper scoring; ECE on raw masked-renorm confidences (no temperature fit here; only e2_laya_zeroshot.py fits T); masked-renormalized
# gate + ECE + risk-coverage on TEST. Stage A extracts features once (npz), Stage B trains.
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

OUT = Path(Path(__file__).resolve().parents[1], "experiments", "real")
LOCAL = str(Path(Path(__file__).resolve().parents[1], "experiments", "real", "models", "laya"))
GOALS = {
    "detection": "detect-stage: gather log evidence for the alert and judge triage disposition",
    "protection": "protect-stage: execute protective handling and record summary",
    "response": "response-stage: investigate attack path and act, record response report",
    "recognition": "recognize-stage: profile entities and record recognition report",
    "dlp-dispose": "dispose-stage: handle DLP alert and record disposal summary",
    "QwenPaw_QA_Agent_0.2": "answer the user question",
    "unknown": "complete the current task",
}

def load_split(name):
    return [json.loads(l) for l in (OUT / f"e2_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

tr_full, va, te = load_split("train"), load_split("val"), load_split("test")
# GPU path: full train split (the 6k subsample era is over; train_sub_idx.json retired)
tr = tr_full
print(f"train rows: {len(tr)}", flush=True)
meta = json.loads((OUT / "e2_meta.json").read_text(encoding="utf-8"))
rows = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
trace_ep = {r["trace"]: r for r in rows}

def state_text(rec):
    ep = trace_ep[rec["trace"]]
    goal = ep["agent"]
    si = rec["t"]["step_idx"]
    chain = ">".join(x.split("__")[-1] for x in ep["labels"][:si + 1][-8:])
    return f"<{goal}> {chain} | next?"

FEAT = OUT / "feats_laya_cls.npz"
PARTS = OUT / "feat_parts"
PARTS.mkdir(exist_ok=True)
CHUNK = 512

def extract():
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    from laya.agent import Agent
    agent = Agent(LOCAL, device=DEV)
    model, tok = agent.model, agent.tok
    model.eval()
    feats = {}
    with torch.inference_mode():
        for tag, recs in (("train", tr), ("val", va), ("test", te)):
            texts = [state_text(r) for r in recs]
            X = []
            t0 = time.perf_counter()
            for i in range(0, len(texts), CHUNK):
                rows_f = []
                for j in range(i, min(i + CHUNK, len(texts)), 128):
                    batch = tok(texts[j:j + 128], padding=True, truncation=True, max_len=64,
                                return_tensors="pt").to(DEV)
                    h = model.encoder(**batch).last_hidden_state
                    rows_f.append(h[:, 0].float().cpu().to(torch.float16))
                arr = torch.cat(rows_f).numpy()
                np.save(PARTS / f"{tag}_{i:05d}.npy", arr)   # provenance checkpoints
                X.append(arr)
                print(f"[{DEV}][{tag}] {min(i + CHUNK, len(texts))}/{len(texts)} {time.perf_counter() - t0:.1f}s", flush=True)
            feats[tag] = np.concatenate(X)
            print(tag, feats[tag].shape, flush=True)
    np.savez_compressed(FEAT, **feats)

if not FEAT.exists():
    extract()
    if int(os.environ.get("N_SHARD", "1")) > 1:
        raise SystemExit(0)   # shard pass written its parts; run assemble_feats.py after both

F = np.load(FEAT)
action_vocab = meta["action_vocab"]
edge_vocab = meta["edge_vocab"]
STOP = meta["stop"]
A2I = {**action_vocab, STOP: len(action_vocab)}
# edge option/target IDs are "e####" strings (values of edge_vocab), key them as such
_edge_ids = sorted(set(edge_vocab.values()))
E2I = {eid: i for i, eid in enumerate(_edge_ids)}
E2I["<stop_edge>"] = len(E2I)
E2I["<unk_edge>"] = len(E2I) + 1

def targets(recs, edge_mode):
    return np.array([E2I.get(r["target_edge_id"], E2I["<unk_edge>"]) if edge_mode else A2I[r["target"]]
                     for r in recs])

def opt_lists(recs, edge_mode):
    out = []
    for r in recs:
        keys = r["option_edge_ids"] if edge_mode else r["options"]
        vocab = E2I if edge_mode else A2I
        unk = vocab["<unk_edge>"] if edge_mode else A2I[STOP]
        out.append([vocab.get(k, unk) for k in keys])
    return out

class Probe(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)

    def forward(self, x):
        return self.fc(x)

def run(edge_mode, objective, seed, epochs=90):
    torch.manual_seed(seed)
    k = len(E2I) if edge_mode else len(A2I)
    tag = "edge" if edge_mode else "action"
    Xtr, ytr = F["train"], targets(tr, edge_mode)
    yva = targets(va, edge_mode)
    yte = targets(te, edge_mode)
    opt_te = opt_lists(te, edge_mode)
    opt_va = opt_lists(va, edge_mode)
    in_space = np.array([r["target_in_opts"] for r in te])
    m = Probe(Xtr.shape[1], k)
    optz = torch.optim.Adam(m.parameters(), lr=5e-3)
    Xt = torch.tensor(Xtr.astype(np.float32))
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xtr)
    for ep in range(epochs):
        order = np.random.RandomState(seed + ep).permutation(n)
        for i in range(0, n, 512):
            b = order[i:i + 512]
            logits = m(Xt[b])
            p = torch.softmax(logits, 1)
            ce = nn.functional.cross_entropy(logits, yt[b])
            oh = torch.zeros_like(p).scatter_(1, yt[b].unsqueeze(1), 1.0)
            brier = ((p - oh) ** 2).sum(1).mean()
            loss = ce if objective == "ce" else ce + brier
            optz.zero_grad(); loss.backward(); optz.step()
    m.eval()
    logits = m(torch.tensor(F["test"].astype(np.float32))).detach().numpy()
    n = len(yte)
    correct = 0; confs = []; cors = []; rows_ = []
    for i in range(n):
        opts = opt_te[i]
        gl = logits[i][opts]
        gl = gl - gl.max()
        p = np.exp(gl); p /= p.sum()
        j = int(p.argmax())
        ok = (opts[j] == yte[i])
        correct += ok
        cors.append(ok); confs.append(float(p[j])); rows_.append((float(p[j]), ok))
    rows_.sort(key=lambda r: -r[0])
    ca = np.array(confs); co = np.array(cors, dtype=float)
    ece = 0.0
    for lo in range(15):
        hi = (lo + 1) / 15
        mm = (ca > lo / 15) & (ca <= hi) if lo else (ca >= 0) & (ca <= hi)
        if mm.sum():
            ece += mm.mean() * abs(ca[mm].mean() - co[mm].mean())
    def risk_at(cov):
        kk = max(1, int(round(cov * n)))
        return round(sum(1 for c, o in rows_[:kk] if not o) / kk, 5)
    isub = co[in_space]
    return {
        "masked_acc": round(correct / n, 4),
        "acc_declared": round(float(isub.mean()), 4) if in_space.any() else None,
        "ece15": round(float(ece), 4),
        "risk_at_cov": {str(c): risk_at(c) for c in (0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0)},
    }

results = {}
for edge_mode, tag in ((False, "action"), (True, "edge")):
    for objective in ("ce", "cebrier"):
        stats = [run(edge_mode, objective, s) for s in range(5)]
        agg = {}
        for k in ("masked_acc", "acc_declared", "ece15"):
            vs = [s[k] for s in stats if s[k] is not None]
            agg[k] = {"mean": round(float(np.mean(vs)), 4), "std": round(float(np.std(vs)), 4)}
        agg["risk_at_cov_mean"] = {
            c: round(float(np.mean([s["risk_at_cov"][c] for s in stats])), 5) for c in stats[0]["risk_at_cov"]
        }
        agg["per_seed"] = [{k: s[k] for k in ("masked_acc", "acc_declared", "ece15")} for s in stats]
        results[f"{tag}_layaprobe_{objective}"] = agg
        print(tag, objective, agg["masked_acc"], agg["acc_declared"], agg["ece15"], flush=True)

(OUT / "e2_layaprobe_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved ->", OUT / "e2_layaprobe_results.json")