# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Math-Verify scoring, run in grader processes with a hard timeout.

`MathVerifyPool` runs this file as a script in each grader process, so it must not
import torchtitan: importing `torchtitan.rl` loads vLLM (~9 s and ~1 GB per process).
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import connection

from math_verify import parse, verify

logger = logging.getLogger(__name__)

_BOXED_START = r"\boxed{"
# The thin space in `10\,002`; Math-Verify reads it as 10 + 002 = 12.
_THIN_SPACE_IN_NUMBER = re.compile(r"(?<=\d)\\,(?=\d{3}(?!\d))")
# A word answer like `indonesian`. 1-2 letters stay math: `ab`, or a segment `CD`.
_WORD = re.compile(r"\s*[A-Za-z]{3,}\s*")
# Spaces between letters, as in `D A E C B`, which would otherwise skip the word check.
_SPACING = re.compile(r"\s+|\\[,;:! ]")
# A LaTeX command or a non-letter, dropped to read the letters of an answer.
_NON_LETTER = re.compile(r"\\[A-Za-z]+|[^A-Za-z]")
# From <sys/prctl.h>: signal this process when the thread that started it exits.
_PR_SET_PDEATHSIG = 1


class MathVerifyPool:
    """Score answers in grader processes, so an answer that hangs Math-Verify cannot
    stall the caller.

    Each grader process (a "worker" below) scores one answer at a time. A worker that
    gives no score within `timeout_seconds` is killed, the answer scores 0, and the next
    score starts a new worker. Pool threads, not the caller's event loop, wait on them.

    Why processes: for `\\boxed{2000^{2000^{2000}}}`, sympy computes 2000**(2000**2000)
    in one C call that never finishes and holds the GIL. In-process:
    1. Every other thread stops, including the event loop.
    2. A thread timeout (`PyThreadState_SetAsyncExc`) fires only between bytecodes.
    3. Math-Verify's SIGALRM timeout needs the main thread; rollout workers score on a
       monarch actor thread.

    Example:
        pool = MathVerifyPool(num_processes=4, timeout_seconds=5.0)
        await pool.score(r"Answer: \\boxed{34}", "34")  # 1.0
        await pool.score(r"Answer: \\boxed{2000^{2000^{2000}}}", "34")  # 0.0 after 5 s
    """

    def __init__(self, *, num_processes: int, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        # Each thread drives at most one worker, started on its first score.
        self._executor = ThreadPoolExecutor(
            max_workers=num_processes, thread_name_prefix="math_verify"
        )
        self._thread_local = threading.local()

    async def score(self, response: str, ground_truth: str) -> float:
        """Return `score_math_response(response, ground_truth)`, or 0.0 on timeout."""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._score_in_worker, response, ground_truth
        )

    def _score_in_worker(self, response: str, ground_truth: str) -> float:
        worker = getattr(self._thread_local, "worker", None)
        # Start a worker on this thread's first score, or after its last one exited.
        if worker is None or worker.poll() is not None:
            worker = self._thread_local.worker = _start_worker()

        worker.stdin.write(json.dumps([response, ground_truth]) + "\n")
        worker.stdin.flush()
        # `connection.wait` uses poll(); select() rejects fds >= 1024.
        if connection.wait([worker.stdout], timeout=self._timeout_seconds):
            # An empty line means the worker died mid-score; treat it like a timeout.
            score_line = worker.stdout.readline()
            if score_line:
                return float(score_line)

        worker.kill()
        worker.wait()
        # TODO: count these per step as a metric; today they only show in this warning.
        logger.warning(
            "Math-Verify worker gave no score within %s seconds, or died; killed it and "
            "assigned zero reward. Response tail: %r",
            self._timeout_seconds,
            response[-200:],
        )
        return 0.0


def score_math_response(response: str, ground_truth: str) -> float:
    """Score the final `\\boxed{}` expression with Math-Verify, in this process.

    A `\\,` before a 3-digit group is dropped, so `10\\,002` matches `10002`, not `12`.
    If answer and gold are each a 3+ letter word, they compare as lowercase strings, and
    an answer whose letters reorder a word gold scores 0.
    No timeout: on an event loop, use `MathVerifyPool.score` instead.

    Args:
        response: Model response containing a boxed final answer.
        ground_truth: Expected answer from the dataset.

    Example:
        score_math_response(r"work\nAnswer: \boxed{34}", "34")  # 1.0
        score_math_response(r"\\boxed{eat}", "tea")  # 0.0
    """
    prediction = _last_boxed_expression(response)
    if prediction is None:
        return 0.0

    prediction = _THIN_SPACE_IN_NUMBER.sub("", prediction)
    ground_truth = _THIN_SPACE_IN_NUMBER.sub("", ground_truth)

    # Compare words as strings: Math-Verify reads `tea` as t*e*a, so `eat` matches,
    # and reads the `I` in `Indonesian` as the imaginary unit.
    answer = _SPACING.sub("", prediction[len(_BOXED_START) : -1])
    if _WORD.fullmatch(answer) and _WORD.fullmatch(ground_truth):
        return float(answer.lower() == ground_truth.strip().lower())
    # Math-Verify reads a word gold as a product, so `(DAECB)` or `D\cdot A\cdot E\cdot C\cdot B`
    # would match `EDACB`: reject an answer whose letters reorder the gold's.
    if _WORD.fullmatch(ground_truth):
        letters = _NON_LETTER.sub("", prediction[len(_BOXED_START) : -1]).lower()
        gold_letters = ground_truth.strip().lower()
        if letters != gold_letters and sorted(letters) == sorted(gold_letters):
            return 0.0

    try:
        # Box the gold like the prediction: a bare `2\sqrt{3}` parses as 2, and a
        # bare `(1,2)` or `\pi/4` parses to nothing.
        gold = parse(_BOXED_START + ground_truth + "}", parsing_timeout=None)
        prediction = parse(prediction, parsing_timeout=None)
        return float(bool(gold) and verify(gold, prediction, timeout_seconds=None))
    except Exception:
        # Model output is untrusted; malformed LaTeX produces a zero reward
        # rather than failing the training loop.
        return 0.0


def _last_boxed_expression(text: str) -> str | None:
    """Return the last complete `\\boxed{...}` expression."""
    start = text.rfind(_BOXED_START)
    if start == -1:
        return None

    answer_start = start + len(_BOXED_START)
    depth = 1
    for index, char in enumerate(text[answer_start:], start=answer_start):
        depth += (char == "{") - (char == "}")
        if depth == 0:
            return text[start : index + 1]
    return None


def _start_worker() -> subprocess.Popen[str]:
    """Start a process running `_serve`; return once it is ready to score."""
    worker = subprocess.Popen(
        # -I: ignore PYTHONPATH and user site-packages, so no parent sitecustomize loads.
        [sys.executable, "-I", __file__],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    # Wait out the interpreter start, so it does not count against the first timeout.
    first_line = worker.stdout.readline()
    if first_line != "ready\n":
        # Kill first: a worker that printed something else is alive, and wait() would hang.
        worker.kill()
        raise RuntimeError(
            f"Math-Verify worker failed to start (exit code {worker.wait()}); "
            f"first stdout line: {first_line!r}"
        )
    return worker


def _serve() -> None:
    """Worker loop: read `[response, ground_truth]` JSON lines, write one score per line."""
    # SIGKILL this worker if its parent dies: a stuck score never sees stdin close.
    ctypes.CDLL(None).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    # Scores get a private copy of fd 1, and fd 1 now goes to stderr, so no library
    # write, Python or C, lands between scores (e.g. ANTLR's version warning).
    scores_out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    print("ready", file=scores_out, flush=True)
    for line in sys.stdin:
        response, ground_truth = json.loads(line)
        print(score_math_response(response, ground_truth), file=scores_out, flush=True)


if __name__ == "__main__":
    _serve()
