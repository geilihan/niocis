# Shared fold rules (§3) -- the ONE implementation, side-effect free so it can
# be imported by any analysis without touching frozen artifacts.
# parse_real_corpus.py produced the frozen SOC artifacts with this fold;
# pub_e1.py folds the public corpora with the same function.
#
# Fold semantics (verbatim from the original): "back" targets an
# already-visited label (visited-label revisit); the returning edge
# (prev -> label) is the recovery back-edge E- candidate. A retry is a
# loop event only: the repeated label is the last kept label (no new
# occurrence kept). Re-keep the label on "back" so downstream decisions
# start from the revisited state.


def fold(label_seq):
    """Return per-step records: kept advances + loop events (retry|back)."""
    kept = []          # kept label sequence
    events = []        # (kind, from_kept_idx, label) full step log
    for lab in label_seq:
        if kept and lab == kept[-1]:
            events.append({"kind": "retry", "label": lab, "prev": kept[-1]})
        elif lab in kept:
            events.append({"kind": "back", "label": lab, "prev": kept[-1]})
            kept.append(lab)  # revisit keeps a new occurrence as the current frame
        else:
            events.append({"kind": "advance", "label": lab, "prev": kept[-1] if kept else None})
            kept.append(lab)
    return kept, events