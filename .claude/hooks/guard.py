#!/usr/bin/env python3
"""Claude Code PreToolUse hook: blocks the Bash commands that cost money or leak data in this repo.

Reads the tool call as JSON on stdin; exit 2 with a message on stderr blocks the call (the message
goes back to Claude). Each rule names the loss that made it (docs/runpod-troubleshooting.md).
An explicit owner decision can pass a rule by putting the marker ALLOW_<RULE> in the command,
e.g. `ALLOW_EXPENSIVE_GPU=1 gh workflow run ...`.
"""
import json
import re
import sys


def block(rule, why):
    print(f"Blocked by .claude/hooks/guard.py [{rule}]: {why} "
          f"(see docs/runpod-troubleshooting.md; the owner can allow it with ALLOW_{rule}).", file=sys.stderr)
    sys.exit(2)


def main():
    try:
        call = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return
    if call.get("tool_name") != "Bash":
        return
    cmd = (call.get("tool_input") or {}).get("command", "")
    allowed = lambda rule: f"ALLOW_{rule}" in cmd  # noqa: E731

    # secrets: the RunPod API is called by the workflows only; a pod listing carries the Kaggle token
    if re.search(r"api\.runpod\.(io|ai)|runpodctl\s+(get|pod)", cmd) and not allowed("RUNPOD_API"):
        block("RUNPOD_API", "do not call the RunPod API from here; a pod listing prints the Kaggle token")
    if re.search(r"gh\s+(secret|variable)\s+(list|get)|printenv|\benv\b\s*\|", cmd) and "KAGGLE" in cmd.upper() \
            and not allowed("SECRETS"):
        block("SECRETS", "never print secrets or environments")

    if "gh workflow run" in cmd or "/dispatches" in cmd:
        # expensive cards: the CPU generator makes a 4090 wait for data (Nov 16: ~5 USD lost)
        gpus = re.findall(r"gpu=(\"[^\"]*\"|'[^']*'|\S+)", cmd)
        if any(re.search(r"4090|L40|A100|H100|H200|A6000|6000 Ada", g, re.I) for g in gpus) \
                and not allowed("EXPENSIVE_GPU"):
            block("EXPENSIVE_GPU", "cheap cards only (<= 0.5 USD/h)")
        # GitHub kills jobs at 355 min: a longer job loses its artifact (Nov 19: rowheavy3d)
        m = re.search(r"max_hours=[\"']?([0-9.]+)", cmd)
        if m and "gpu-train" in cmd and float(m.group(1)) > 5.8 and not allowed("LONG_RUN"):
            block("LONG_RUN", "max_hours > 5.8 runs into GitHub's 6 h limit")
        m = re.search(r"--minutes\s+([0-9.]+)", cmd)
        if m and float(m.group(1)) > 290 and not allowed("LONG_RUN"):
            block("LONG_RUN", "--minutes > 290 leaves no time for setup before the 340 min follower limit")
        # account limit 64 vCPU: more pods only queue (Nov 20: three runs failed)
        m = re.search(r"pods=[\"']?(\d+)", cmd)
        if m and int(m.group(1)) > 2 and not allowed("MANY_PODS"):
            block("MANY_PODS", "at most two 32-vCPU pods at a time")
        # downloads on 32 vCPU pay for idle CPUs (Nov 19: ~1.7 USD)
        if "pod-job" in cmd and re.search(r"vcpu=[\"']?(\d+)", cmd):
            v = int(re.search(r"vcpu=[\"']?(\d+)", cmd).group(1))
            if v > 32:
                block("MANY_PODS", "pod-job above 32 vCPU")
        if "gpu-stop-all" in cmd and not allowed("STOP_ALL"):
            block("STOP_ALL", "gpu-stop-all kills every training pod of the account, also parallel runs")

    # competition data never goes into git
    if re.search(r"git\s+add", cmd) and re.search(r"physionet/data|/sig/|realstrips/|visual_data|\.npz\b|kaggle/train",
                                                   cmd) and not allowed("DATA_IN_GIT"):
        block("DATA_IN_GIT", "competition and benchmark data never go into the repository")


if __name__ == "__main__":
    main()
