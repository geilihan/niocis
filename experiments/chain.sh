#!/bin/bash
# LW2 remote experiment chain - all model/experiment compute runs inside the
# container on GPUs 0,1 only; all paths confined to /datagroup/lunwen2 (mount /lw2).
set -e
IMG=nvcr.io/nvidia/pytorch:26.03-py3

docker rm -f lw2 >/dev/null 2>&1 || true
docker run -d --name lw2 --gpus '"device=0,1"' --shm-size 16g \
  -v /datagroup/lunwen2:/lw2 -w /lw2 "$IMG" sleep infinity
echo "[boot] container lw2 up (GPU 0,1 only, /lw2 confined)"

echo "[deps] installing via domestic mirror (aliyun); laya 0.3.21 via proxied pypi (aliyun only carries 0.3.5)"
docker exec lw2 bash -lc "pip config --global set global.index-url https://mirrors.aliyun.com/pypi/simple/; pip install -q 'laya==0.3.21' --index-url https://pypi.org/simple || pip install -q 'laya==0.3.21' --index-url https://pypi.org/simple -i https://mirrors.aliyun.com/pypi/simple/ || pip install -q 'laya==0.3.5'; pip install -q scikit-learn matplotlib"
docker exec lw2 python -c "import torch,laya,sklearn,matplotlib; print('[deps] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'devs')"

for s in parse_real_corpus.py e1_real.py e2_build.py e2_train_torch.py gpu_latency_smoke.py e2_laya_probe.py e2_laya_zeroshot.py e3_replay.py e4_rsi.py e8_slices.py e8_stats.py make_tab_corpus.py render_figs_tabs_real.py; do
  echo "[run] $s"
  docker exec lw2 python /lw2/code/$s
done
echo ALL_DONE