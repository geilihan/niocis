#!/usr/bin/env python
"""Verify this checkout byte for byte against MANIFEST.sha256.

Every published file is compared with both its recorded size and its recorded
sha256, so a line-ending conversion (or any other edit) shows up as a mismatch.
Exit code is 0 only when every entry matches and the tree is fully covered.

    python experiments/verify_manifest.py [--root .]
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

ENTRY = re.compile(r"^([0-9a-f]{64})\s+(.+?)\s+\((\d+)\s*B\)$")


def main() -> int:
    ap = argparse.ArgumentParser(description="verify the published tree against MANIFEST.sha256")
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    mf = root / "MANIFEST.sha256"
    if not mf.is_file():
        print("MANIFEST.sha256 not found under %s" % root)
        return 2

    entries = {}
    unparsable = []
    for line in mf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = ENTRY.match(line.rstrip("\r"))
        if not m:
            unparsable.append(line[:80])
            continue
        entries[m.group(2)] = (m.group(1), int(m.group(3)))

    ok = missing = size_bad = sha_bad = 0
    for rel, (want_sha, want_size) in sorted(entries.items()):
        p = root / rel
        if not p.is_file():
            print("MISSING   %s" % rel)
            missing += 1
            continue
        raw = p.read_bytes()
        if len(raw) != want_size:
            print("SIZE      %s  manifest=%d file=%d (%+d)" % (rel, want_size, len(raw), len(raw) - want_size))
            size_bad += 1
            continue
        got = hashlib.sha256(raw).hexdigest()
        if got != want_sha:
            print("SHA256    %s  manifest=%s file=%s" % (rel, want_sha[:16], got[:16]))
            sha_bad += 1
            continue
        ok += 1

    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    extra = sorted(present - set(entries) - {"MANIFEST.sha256"})
    for rel in extra:
        print("UNLISTED  %s (present but not in MANIFEST.sha256)" % rel)
    for line in unparsable:
        print("UNPARSED  %s" % line)

    print(
        "entries %d  ok %d  missing %d  size-mismatch %d  sha-mismatch %d  unlisted %d  unparsable %d"
        % (len(entries), ok, missing, size_bad, sha_bad, len(extra), len(unparsable))
    )
    bad = missing + size_bad + sha_bad + len(extra) + len(unparsable)
    print("MANIFEST %s" % ("OK: every published file matches" if bad == 0 else "FAILED: %d problem(s)" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
