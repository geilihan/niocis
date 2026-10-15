# Peek at the FIRST bytes of each tool dataset to pin the row schema before
# wiring the parser (streamed reads; the xlam json alone is ~200MB).
import gzip
import urllib.request

PROXY = ""  # set HTTP_PROXY if your network requires a proxy
U = "ariadne-repl/1.0"


def peek(url, n=1500, direct=True):
    h = urllib.request.ProxyHandler({} if direct else {"http": PROXY, "https": PROXY})
    op = urllib.request.build_opener(h)
    req = urllib.request.Request(url, headers={"User-Agent": U, "Range": f"bytes=0-{n}"})
    try:
        with op.open(req, timeout=30) as r:
            return r.read(n + 1)
    except Exception as e:
        return f"<ERR {type(e).__name__} {getattr(e, 'code', '')}>".encode()


TARGETS = {
    "ab-level1-sample": "https://gh-proxy.com/https://raw.githubusercontent.com/"
                        "AlibabaResearch/DAMO-ConvAI/main/api-bank/"
                        "lv1-lv2-samples/level-1-given-desc/Calculator-level-1-1.jsonl",
    "ab-level2-sample": "https://gh-proxy.com/https://raw.githubusercontent.com/"
                        "AlibabaResearch/DAMO-ConvAI/main/api-bank/"
                        "lv1-lv2-samples/level-2-toolsearcher/BookHotel-level-3-1.jsonl",
    "xlam-head": "https://hf-mirror.com/datasets/Salesforce/xlam-function-calling-60k/"
                 "resolve/main/xlam_function_calling_60k.json",
    "toolllama-eval-head": "https://hf-mirror.com/datasets/Adorg/ToolBench/"
                           "resolve/main/toolllama_G123_dfs_eval.json",
}
for tag, url in TARGETS.items():
    body = peek(url)
    print(f"----- {tag} ({len(body)}B) -----")
    print(body[:1400].decode("utf-8", "replace"))
print("POKE_DONE")