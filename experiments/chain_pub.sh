#!/bin/bash
# Public-corpus chain (item #2): fetch -> xes2jsonl -> pub_e1 -> render_pub.
set -u
export http_proxy="${http_proxy:-}"   # set this if your network requires a proxy
export https_proxy="${https_proxy:-}"   # set this if your network requires a proxy
export no_proxy="10.0.0.0/8,localhost,127.0.0.1,*.aliyun.com,*.aliyuncs.com"
for s in pub_fetch.py pub_xes2jsonl.py pub_e1.py render_pub.py; do
  echo "[run] $s"
  docker exec -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" lw2 python /lw2/code/$s || echo "[fail] $s"
done
echo PUB_DONE