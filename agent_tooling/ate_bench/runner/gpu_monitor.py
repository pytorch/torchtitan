"""Measure 'active GPU time' during an agent session by sampling nvidia-smi.

ATE-Bench reports *active GPU time* for the operate/profile and new-feature tasks
(the GPU busy time the agent's training runs consume). We approximate it by
polling per-GPU utilization at a fixed interval and integrating the busy time:

    active_gpu_time_s = sum over GPUs of (interval * count of samples where
                                          utilization > busy_threshold_pct)

i.e. GPU-seconds of busy time, summed across all visible GPUs. This counts a GPU
as "active" while it is doing work, which is the quantity that differs between
frameworks for the same task. If nvidia-smi is unavailable (no GPU), the monitor
degrades gracefully and reports ``active_gpu_time_s = None``.

Usage (around the agent subprocess):

    mon = GpuMonitor().start()
    ... run the agent ...
    stats = mon.stop()   # {'active_gpu_time_s': float|None, 'wall_s': float, ...}
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field


def _sample_utilization() -> list[int] | None:
    """Return per-GPU utilization percentages, or None if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    vals = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if line:
            try:
                vals.append(int(float(line)))
            except ValueError:
                pass
    return vals or None


@dataclass
class GpuMonitor:
    interval_s: float = 1.0
    busy_threshold_pct: int = 5
    _busy_gpu_seconds: float = 0.0
    _samples: int = 0
    _available: bool = False
    _t0: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            util = _sample_utilization()
            if util is not None:
                self._available = True
                busy = sum(1 for u in util if u > self.busy_threshold_pct)
                self._busy_gpu_seconds += busy * self.interval_s
                self._samples += 1
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuMonitor":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)
        wall = time.time() - self._t0
        return {
            "active_gpu_time_s": (self._busy_gpu_seconds if self._available else None),
            "wall_s": wall,
            "samples": self._samples,
            "interval_s": self.interval_s,
            "busy_threshold_pct": self.busy_threshold_pct,
            "gpu_monitor_available": self._available,
        }


def main(argv: list[str] | None = None) -> int:
    """Standalone: monitor a wrapped command, print active GPU time."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run and monitor")
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("provide a command after --")
    mon = GpuMonitor(interval_s=args.interval).start()
    rc = subprocess.run(cmd).returncode
    print(json.dumps(mon.stop(), indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
