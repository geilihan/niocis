# One-shot probe: resolve REAL file listings for tool datasets via mirror APIs
# (hf-mirror.com, gh-proxy.com). Run after guessed URLs 404'd; feed results back
# into pub_tool_fetch.py candidates. Prints a summary + drops raw JSON in cache/.
import json
import urllib.request
from pathlib import Path

CACHE = Path("/lw2/cache") if Path("/lw2").exists() else Path(Path(__file__).resolve().parents[1], "experiments", "real")
PROXY = ""  # set HTTP_PROXY if your network requires a proxy


def get(url, direct=True):
    h = urllib.request.ProxyHandler({} if direct else {"http": PROXY, "https": PROXY})
    op = urllib.request.build_opener(h)
    req = urllib.request.Request(url, headers={"User-Agent": "ariadne-repl/1.0"})
    with op.open(req, timeout=30) as r:
        return r.read()


JOBS = {
    "ab_l1": "https://gh-proxy.com/https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/contents/api-bank/lv1-lv2-samples/level-1-given-desc",
    "ab_l2": "https://gh-proxy.com/https://api.github.com/repos/AlibabaResearch/DAMO-ConvAI/contents/api-bank/lv1-lv2-samples/level-2-toolsearcher",
    "tb_instr": "https://hf-mirror.com/api/datasets/Adorg/ToolBench/tree/main/instruction",
}

out = {}
for tag, url in JOBS.items():
    for direct in (True, False):
        try:
            body = get(url, direct=direct)
            out[tag] = json.loads(body)
            (CACHE / f"probe_{tag}.json").write_bytes(body)
            print(f"[ok] {tag} ({'direct' if direct else 'proxy'}) {len(body)}B")
            break
        except Exception as e:
            print(f"[x]  {tag} ({'direct' if direct else 'proxy'}) {type(e).__name__} {getattr(e, 'code', '')}")

if "ab_l1" in out and isinstance(out["ab_l1"], list):
    print("AB level-1-given-desc:", [(x.get("name"), x.get("size")) for x in out["ab_l1"]][:30])
if "ab_l2" in out and isinstance(out["ab_l2"], list):
    print("AB level-2-toolsearcher:", [(x.get("name"), x.get("size")) for x in out["ab_l2"]][:30])
if "tb_instr" in out and isinstance(out["tb_instr"], list):
    print("TB instruction/:", [(x.get("path"), x.get("size")) for x in out["tb_instr"]][:20])
try:
    old = json.loads((CACHE / "probe_tb_adorg.json").read_text())
    print("tb_adorg root sizes:", [(x.get("path"), x.get("size")) for x in old])
except Exception as e:
    print("size peek failed", type(e).__name__)
print("PROBE_DONE")