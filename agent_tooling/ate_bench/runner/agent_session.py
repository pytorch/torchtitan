"""Shared headless-agent session runner with GPU-time monitoring.

Used by run_operate.py and run_feature.py. Invokes the fixed agent headless,
captures the stream-json transcript, parses ATE effort metrics, and measures
active GPU time across the session.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as metrics_mod  # noqa: E402
from gpu_monitor import GpuMonitor  # noqa: E402


def run_session(
    prompt: str,
    cwd: str | Path,
    transcript_path: str | Path,
    *,
    allowed_tools: list[str],
    disallowed_tools: tuple[str, ...] | list[str] = (),
    permission_mode: str | None = None,
    model: str | None = None,
    timeout: int = 3600,
    monitor_gpu: bool = True,
):
    """Run one agent session. Returns (TaskMetrics|None, returncode, stderr, gpu_stats)."""
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
    if allowed_tools:
        cmd += ["--allowedTools", *allowed_tools]
    if disallowed_tools:
        cmd += ["--disallowedTools", *list(disallowed_tools)]
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if model:
        cmd += ["--model", model]

    mon = GpuMonitor().start() if monitor_gpu else None
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, cwd=str(cwd), timeout=timeout
        )
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        def _s(x):
            if x is None:
                return ""
            return x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray)) else x
        rc, stdout, stderr = 124, _s(exc.stdout), _s(exc.stderr) + "\n[timeout]"
    gpu = mon.stop() if mon else {}

    tp = Path(transcript_path)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(stdout or "", encoding="utf-8")
    if stderr:
        tp.with_suffix(".stderr.log").write_text(stderr, encoding="utf-8")

    m = None
    if (stdout or "").strip():
        try:
            m = metrics_mod.parse_transcript(tp)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! failed to parse transcript: {exc}", file=sys.stderr)
    # Fold active GPU time into the metrics object.
    if m is not None and gpu:
        m.active_gpu_time_s = gpu.get("active_gpu_time_s")
    return m, rc, stderr, gpu
