# Remote environment survey via paramiko (no local installs, no C: writes).
import paramiko

import os

HOST = os.environ.get("ARIADNE_HOST", "")
USER = os.environ.get("ARIADNE_USER", "root")
PORT = int(os.environ.get("ARIADNE_PORT", "22"))
PW = os.environ.get("ARIADNE_PW", "")  # R-146: never hardcode a credential in a source file

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)

cmds = [
    "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv",
    "python3 --version 2>&1 || python --version 2>&1",
    "python3 -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())\" 2>&1 | tail -1",
    "df -h / /tmp 2>/dev/null | tail -3",
    "df -h | head -8",
    "nproc",
    "pip3 --version 2>&1 | head -1",
    "ls /data 2>/dev/null | head -4",
    "echo HOME=$HOME",
]
for c in cmds:
    _, out, err = cli.exec_command(c, timeout=60)
    o = out.read().decode(errors="replace").strip()
    e = err.read().decode(errors="replace").strip()
    print(f"$ {c}")
    if o:
        print("  " + "\n  ".join(o.splitlines()[:10]))
    if e:
        print("  [err] " + "\n  ".join(e.splitlines()[:3]))
cli.close()