# NIOCIS --- typed-decision next-hop routing with node-conditioned option spaces

Research code and the public, pseudonymized derived corpus for the paper
*NIOCIS: Typed-Decision Next-Hop Routing with Node-Conditioned Option Spaces for Tool-Use Agents*
(preprint: `TODO(author): arXiv ID / DOI after posting`).

The pipeline folds production agent traces into a loop-collapsed typed action graph and
answers "which edge in `A(v)` leads to task completion?" per state with a lightweight
typed-decision router, escalating only uncertain cases to an LLM. Executed walks flow back
as gate-checked updates.

## Naming

The system is called **NIOCIS** in the paper. Several scripts keep the project's historical
`ARIADNE_*` identifier prefix (for example `ARIADNE_LLM_KEY`, `ARIADNE_SSH_FILE`, and a few
generated user-agent strings). They are left unchanged on purpose, so that the released code
stays byte-identical to the frozen experiment records behind the reported numbers.

## Redactions

Six released fetch/bootstrap scripts (`experiments/chain.sh`, `experiments/chain_pub.sh`,
`experiments/poke_samples.py`, `experiments/probe_hf.py`, `experiments/pub_fetch.py`,
`experiments/pub_tool_fetch.py`) carried an internal proxy hostname when they were run. That host
is not published here: `chain_pub.sh` now takes the proxy from `http_proxy` / `https_proxy`
(unset by default, so the script runs direct), the Python fetchers use an empty `PROXY` default,
and `chain.sh` drops the `--proxy` argument that its mirror fallback did not need. Set your own
proxy if your network requires one. This is the only staged-only edit made by the release script:
every other released file is byte-identical to its frozen source, and the sources themselves keep
the historical hostname because the experiment receipts hash the files exactly as they were
executed.

## Layout

| path | what |
|---|---|
| `experiments/` | the released pipeline: corpus assembly, feature assembly, the router/probe/Markov/stream experiments, figure+table renderers, and the release leak gates |
| `experiments/addendum/` | frozen extra-experiment artifacts (E3/E4/E9/E9b option-space arms, latency sweep) |
| `experiments/addendum_src/` | sources for those extra experiments |
| `experiments/e21_release_reproduction.py` | the check that the numbers below are reproducible from `release/` alone |
| `release/` | the public pseudonymized derived corpus (see `release/README.md`) |
| `paper/figures/`, `paper/tables/` | the LaTeX tables and the figure PDFs/PNGs rendered from the frozen artifacts by `render_figs_tabs_real.py` / `render_pub.py` |

## Reproduce the reported corpus-level numbers from the public data only

```bash
python experiments/e21_release_reproduction.py
```

It re-derives, from `release/` alone: 1,615 episodes / 1,611 sessions, the 12,613 / 1,580 /
3,875 train/val/test split, the 1,174 / 245 / 139 / 54 per-slice counts, the held-out
in-space coverage 3,568 of 3,875, the held-out mean declared option space 40.21, the 154-name
catalog (plus the separate stop option), 22,192 tool calls and 22,328 observations, and the
4,124 retry / 2,364 back / 15,704 advance fold totals --- all of which match the paper.
Verification receipt: the paper's Reproducibility statement and `release/RESIDUALS.txt`.

## What is *not* here

- The manuscript's LaTeX sources and PDF. They are kept out of the code release (the paper is
  distributed separately); `paper/tables/` and `paper/figures/` are kept because the generator
  scripts write their output straight into that tree.
- `experiments/real/` --- the internal frozen artifacts listed in the paper's Reproducibility
  statement. They are derived from a production SOC telemetry export and are **not** part of
  this repository; the pseudonymized `release/` bundles are.
- The raw Langfuse export, tool-argument payloads and chat bodies.
- The action-code <-> internal-name mapping and the identifier hash salt (both withheld; see
  `release/README.md`, section *Withheld by design*).
- Model-level reproduction requires the internal feature tensor, so this repository supports
  corpus-level reproduction only.
- `experiments/_ssh.py` and `experiments/check_packaging.py` (private-credential harness and
  internal packaging checker).

Two addendum scripts (`e9_graph_necessity.py`, `e9b_node_prior.py`) keep the author's local
paths because they read the unpublished frozen artifacts; they document method, they do not
run from this repository.

## Verify your copy

Every published file is listed in `MANIFEST.sha256` with its sha256 and its size, and the tree
is pinned byte for byte (`.gitattributes` disables end-of-line conversion), so the listing
validates on any platform and in any clone:

```bash
python experiments/verify_manifest.py
```

It reports every missing file, every size or sha256 mismatch, and every file the manifest does
not cover. Some frozen sources keep CRLF line endings on purpose: they are byte-identical to
the files the experiment records hashed, so they are published unchanged.

## Licence

The pseudonymized derived corpus in `release/` is CC BY 4.0 (see `release/LICENSE.txt`).
The **code** licence is not yet chosen by the author --- see `LICENSE.md`.

## Citation

See `CITATION.cff`.
