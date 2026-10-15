# Fetch PUBLIC tool-use trajectory datasets into PUB/ (item #3).
# Sources VERIFIED via probe_hf.py API listings on 2026-09-29 (the lesson stands:
# never paste recalled URLs):
#   apibank_raw/  API-Bank lv1-lv2 scenario files (AlibabaResearch/DAMO-ConvAI),
#                 one .jsonl = one dialogue session; JSONL turns role in
#                 {User, AI, API}, tool calls are role=="API" turns (api_name)
#   xlam.json     Salesforce/xlam-function-calling-60k single JSON list; rows use
#                 answers:[{name, arguments}]
#   toolbench_*   Adorg/ToolBench toolllama_G123_dfs_{train,eval}.json (dfs chains;
#                 train is 272MB)
# Domestic mirrors first (gh-proxy, hf-mirror), corp proxy only as fallback.
# Records bytes + sha256 in tool_manifest.json.
import hashlib
import json
import signal
import urllib.request
from pathlib import Path

PROXY = ""  # set HTTP_PROXY if your network requires a proxy
_HOST = Path("/lw2/pubdata")
PUB = _host = _HOST if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "数据", "pub")
PUB.mkdir(parents=True, exist_ok=True)

GH_PROXY = "https://gh-proxy.com/"
AB_API_M = GH_PROXY + "https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/contents/api-bank/"
AB_API_D = "https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/contents/api-bank/"
AB_RAW_M = GH_PROXY + "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/api-bank/"
AB_RAW_D = "https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/api-bank/"


def openers():
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    proxied = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    return direct, proxied


def _alarm(signum, frame):
    raise TimeoutError("url watchdog fired")


signal.signal(signal.SIGALRM, _alarm)


def try_get(urls, dest, timeout=120, min_bytes=2000):
    direct, proxied = openers()
    for url in urls:
        needs_proxy = (("zenodo.org" in url) or url.startswith("https://huggingface.co")
                       or url.startswith("https://hf.co")
                       or ("raw.githubusercontent.com" in url and "gh-proxy" not in url)
                       or ("api.github.com" in url and "gh-proxy" not in url))
        op = proxied if needs_proxy else direct
        signal.alarm(240)  # getaddrinfo hangs do not respect socket timeouts
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ariadne-repl/1.0"})
            with op.open(req, timeout=timeout) as r:
                clen = r.headers.get("Content-Length")
                with open(dest, "wb") as f:
                    n = 0
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf); n += len(buf)
            if n < min_bytes:
                dest.unlink(missing_ok=True)
                print(f"[tiny] {url[:80]} ({n}B)")
                continue
            if clen is not None and n != int(clen):
                dest.unlink(missing_ok=True)
                raise IOError(f"truncated: got {n} of {clen} bytes")
            if dest.suffix in (".json", ".jsonl") and n > 32:
                # cheap truncation detector: JSON/JSONL payloads close on } or ]
                with open(dest, "rb") as fh:
                    fh.seek(max(0, n - 32))
                    if not fh.read(32).rstrip().endswith((b"}", b"]")):
                        dest.unlink(missing_ok=True)
                        raise IOError("payload does not close on } or ]")
            print(f"[ok]   {n / 1e6:6.1f}MB  {url[:80]}")
            return url, n
        except Exception as e:
            dest.unlink(missing_ok=True)
            print(f"[fail] {url[:80]} ({type(e).__name__} {getattr(e, 'code', '')})")
        finally:
            signal.alarm(0)
    return None, 0


def github_json(urls):
    direct, proxied = openers()
    for u in urls:
        for op in (direct, proxied):
            try:
                with op.open(urllib.request.Request(u, headers={"User-Agent": "ariadne-repl/1.0"}),
                             timeout=60) as r:
                    return json.load(r)
            except Exception as e:
                print(f"[fail-list] {u[:70]} ({type(e).__name__} {getattr(e, 'code', '')})")
    return []


manifest = {}

# ---- API-Bank: scenario .jsonl files, one dialogue session per file ----------
outdir = PUB / "apibank_raw"
outdir.mkdir(exist_ok=True)
ab_seen = 0
for d in ("lv1-lv2-samples/level-1-given-desc", "lv1-lv2-samples/level-2-toolsearcher"):
    listing = github_json([AB_API_M + d, AB_API_D + d])
    files = [i["name"] for i in listing
             if isinstance(i, dict) and i.get("name", "").endswith(".jsonl")]
    if not files:
        print(f"[miss] api-bank/{d}: listing empty")
    for nm in sorted(files):
        url, n = try_get([AB_RAW_M + d + "/" + nm, AB_RAW_D + d + "/" + nm],
                         outdir / nm, min_bytes=200)
        if url:
            ab_seen += 1
manifest["apibank"] = ({"files": ab_seen, "dir": "apibank_raw"} if ab_seen
                       else {"file": None, "error": "listing/download failed"})
print(f"apibank: {ab_seen} scenario files")

# ---- xlam 60k + ToolBench dfs -----------------------------------------------
MIRRORED = {
    # hf-mirror stalls silently on some resolve URLs; every mirror URL is paired
    # with a proxied direct fetch so the ladder keeps moving (inactivity timeout
    # dropped to 120s in try_get)
    "xlam": ["https://hf-mirror.com/datasets/Salesforce/xlam-function-calling-60k/"
             "resolve/main/xlam_function_calling_60k.json",
             "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/"
             "resolve/main/xlam_function_calling_60k.json"],
    "toolbench_eval": ["https://hf-mirror.com/datasets/Adorg/ToolBench/"
                       "resolve/main/toolllama_G123_dfs_eval.json",
                       "https://huggingface.co/datasets/Adorg/ToolBench/"
                       "resolve/main/toolllama_G123_dfs_eval.json"],
    "toolbench_train": ["https://hf-mirror.com/datasets/Adorg/ToolBench/"
                        "resolve/main/toolllama_G123_dfs_train.json",
                        "https://huggingface.co/datasets/Adorg/ToolBench/"
                        "resolve/main/toolllama_G123_dfs_train.json"],
}
for ds, urls in MIRRORED.items():
    dest = PUB / f"{ds}.json"
    if dest.exists() and dest.stat().st_size > 100000:
        with open(dest, "rb") as fh:
            fh.seek(max(0, dest.stat().st_size - 32))
            tail_closes = fh.read(32).rstrip().endswith((b"}", b"]"))
        if tail_closes:
            old = manifest.get(ds, {}) or {}
            manifest[ds] = {"url": old.get("url", "cached"), "file": dest.name,
                            "bytes": dest.stat().st_size,
                            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
            print(f"[cache] {ds}  {dest.stat().st_size / 1e6:.1f}MB  (reused verified copy)")
            continue
        dest.unlink(missing_ok=True)  # partial/stale copy: refetch
    url, n = try_get(urls, dest, min_bytes=100000)
    if url:
        manifest[ds] = {"url": url, "file": dest.name, "bytes": n,
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}

(PUB / "tool_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print(f"tool datasets fetched: {sorted(manifest)}")
raise SystemExit(0 if manifest else 1)