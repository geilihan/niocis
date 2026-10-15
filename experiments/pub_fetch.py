# Fetch public process-mining event logs (E1 external replication) into PUB/.
# Source ladder per corpus: known direct URLs first, then Zenodo API discovery;
# domestic mirror (gh-proxy) preferred for github raw, corp proxy for everything
# external. Records sha256 + bytes per file in fetch_manifest.json.
import gzip
import hashlib
import html
import json
import os
import re
import signal
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PROXY = ""  # set HTTP_PROXY if your network requires a proxy
# gate on the /lw2 mount point: pubdata may not exist yet on a fresh container,
# and the output dir must never gate itself into the Windows-path fallback
_HOST = Path("/lw2/pubdata")
PUB = _HOST if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
PUB.mkdir(parents=True, exist_ok=True)

GH = "https://raw.githubusercontent.com/pm4py/pm4py-core/master/tests/input_data/"
GM = "https://gh-proxy.com/" + GH  # domestic mirror: first in every ladder
# file names below are verified against the pm4py-core tests/input_data listing;
# 4TU file URLs are extension-less (verified via the figshare-style API, so extension
# is sniffed from content magic, not guessed from the URL)
CORPORA = {
    "receipt": [GM + "receipt.xes", GH + "receipt.xes"],
    "roadtraffic": [GM + "roadtraffic_10000.xes.gz", GH + "roadtraffic_10000.xes.gz",
                    GM + "roadtraffic100traces.xes", GH + "roadtraffic100traces.xes"],
    "helpdesk": [GM + "helpdesk.xes.gz", GH + "helpdesk.xes.gz"],
    "bpi12": [GM + "bpic2012.xes.gz", GH + "bpic2012.xes.gz",
              "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"],
    # same URL x2: single-candidate ladder + mid-stream truncation means one shot
    # is flaky; integrity check fails -> falls through to the next entry
    "bpi17": ["https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c"] * 2,
    "statechart": ["https://data.4tu.nl/file/3ec3daab-c7a4-4e4f-bdc4-02c23d488677/b964dc3a-981c-4259-a82a-3430a5787e32"] * 2,
    # "sepsis" dropped 2026-09-29: zenodo discovery only found a course homework
    # bundle (NNN_add_events.csv scaffolding), not the canonical Mannhardt log --
    # uncertain lineage is worse than a missing row (same call as bpi11)
}
ZENODO_SEARCH = {
    # corpus -> discovery query used when every direct candidate fails
    "helpdesk": "helpdesk event log process mining",
    "bpi12": "BPI Challenge 2012 event log",
}


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": PROXY, "https": PROXY}))


def _alarm(signum, frame):
    raise TimeoutError("url watchdog fired")


signal.signal(signal.SIGALRM, _alarm)


def try_get(url, dest, timeout=60):
    if "gh-proxy.com" in url:
        # domestic mirror: force direct (build_opener() would inherit env proxies)
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        op = opener()
    req = urllib.request.Request(url, headers={"User-Agent": "ariadne-repl/1.0"})
    signal.alarm(180)  # getaddrinfo hangs do not respect socket timeouts
    try:
        with op.open(req, timeout=timeout) as r:
            clen = r.headers.get("Content-Length")
            n = 0
            with open(dest, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    n += len(buf)
            if clen is not None and n != int(clen):
                dest.unlink(missing_ok=True)
                raise IOError(f"truncated: got {n} of {clen} bytes")
            return n
    finally:
        signal.alarm(0)


def zenodo_discover(query):
    api = "https://zenodo.org/api/records?q=" + urllib.parse.quote(query) + "&size=5"
    try:
        with opener().open(urllib.request.Request(api, headers={"User-Agent": "ariadne-repl/1.0"}),
                           timeout=60) as r:
            data = json.load(r)
        for hit in data.get("hits", {}).get("hits", []):
            for f in hit.get("files", []):
                key = f.get("key", "")
                link = (f.get("links") or {}).get("self") or ""
                # match on the KEY: zenodo "self" links end in /content and hide the ext
                if link and re.search(r"\.(xes|zip)(\.gz)?$", key, re.I):
                    return [link]
    except Exception as e:
        print("zenodo discovery failed:", query, type(e).__name__)
    return []


manifest = {}
try:
    OLD = json.loads((PUB / "fetch_manifest.json").read_text(encoding="utf-8"))
except Exception:
    OLD = {}
for name, cands in CORPORA.items():
    done = False
    # reuse an already-verified local copy (integrity checked on arrival): keeps
    # rerun cost near zero and avoids re-fetching 30MB of bpi17 per chain run
    for ex in (".xes", ".gz", ".zip"):
        prev = PUB / f"{name}{ex}"
        if prev.exists():
            try:
                if prev.suffix == ".gz":
                    with gzip.open(prev, "rb") as g:
                        while g.read(1 << 22):
                            pass
                elif prev.suffix == ".zip":
                    with zipfile.ZipFile(prev) as z:
                        if z.testzip() is not None:
                            raise IOError("zip CRC error")
                manifest[name] = {"url": OLD.get(name, {}).get("url", "unknown"),
                                  "file": prev.name, "bytes": prev.stat().st_size,
                                  "sha256": hashlib.sha256(prev.read_bytes()).hexdigest()}
                print(f"[cache] {name}  {prev.stat().st_size / 1e6:.1f}MB  (reused verified copy)")
                done = True
                break
            except Exception:
                prev.unlink(missing_ok=True)
    if done:
        continue
    tried = list(cands) + (zenodo_discover(ZENODO_SEARCH[name]) if name in ZENODO_SEARCH else [])
    for url in tried:
        dest = PUB / f"{name}.part"
        try:
            n = try_get(url, dest)
        except Exception as e:
            print(f"[fail] {name} <- {url.split('?')[0][:70]} "
                  f"({type(e).__name__} {getattr(e, 'code', '')})")
            continue
        if n < 50000:
            print(f"[tiny] {name} <- {url.split('?')[0][:70]} ({n}B) - rejected (stub)")
            dest.unlink(missing_ok=True)
            continue
        # sniff content magic: 4TU links carry no filename extension
        head = dest.read_bytes()[:4]
        ext = ".gz" if head[:2] == b"\x1f\x8b" else (".zip" if head[:2] == b"PK" else ".xes")
        # full-payload integrity before accepting (silent mid-stream truncation
        # is how bpi17 arrived as 17.6 of 29.7MB on 2026-09-29)
        try:
            if ext == ".gz":
                with gzip.open(dest, "rb") as g:
                    while g.read(1 << 22):
                        pass
            elif ext == ".zip":
                with zipfile.ZipFile(dest) as z:
                    if z.testzip() is not None:
                        raise IOError("zip CRC error")
        except Exception as e:
            print(f"[corrupt] {name} <- {url.split('?')[0][:70]} ({type(e).__name__})")
            dest.unlink(missing_ok=True)
            continue
        final = PUB / f"{name}{ext}"
        dest.rename(final)
        digest = hashlib.sha256(final.read_bytes()).hexdigest()
        manifest[name] = {"url": url, "file": final.name, "bytes": n, "sha256": digest}
        print(f"[ok]   {name}  {n / 1e6:.1f}MB  {url.split('?')[0][:70]}")
        done = True
        break
    if not done:
        manifest[name] = {"file": None, "error": "all candidates failed"}
        print(f"[MISS] {name}: every candidate failed")

(PUB / "fetch_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
got = [k for k, v in manifest.items() if v.get("file")]
print(f"fetched {len(got)}/{len(CORPORA)}: {got}")
raise SystemExit(0 if len(got) >= 3 else 1)