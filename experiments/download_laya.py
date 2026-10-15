# Robust Laya checkpoint downloader: retries with resume across hub endpoints.
import os
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPOS = ["convaiinnovations/laya", "convaiinnovations/laya-multilingual"]
FILES = ["model.safetensors", "rl_agent_config.json", "config.json"]
ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

for repo in REPOS[:1]:  # english checkpoint first; multilingual later if needed
    for fname in FILES:
        for attempt in range(1, 19):
            ep = ENDPOINTS[(attempt - 1) % 2]
            os.environ["HF_ENDPOINT"] = ep
            try:
                p = hf_hub_download(repo, fname, resume_download=True)
                print(f"OK  {repo}/{fname} via {ep} -> {p}")
                break
            except Exception as e:
                msg = str(e)[:120].replace("\n", " ")
                print(f"try{attempt:02d} fail {repo}/{fname} via {ep}: {msg}")
                time.sleep(3)
        else:
            print(f"GIVEUP {repo}/{fname}")
            raise SystemExit(1)
print("all files downloaded")