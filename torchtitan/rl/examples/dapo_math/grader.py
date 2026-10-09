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

from math_verify import LatexExtractionConfig, LatexNormalizationConfig, parse, verify
from sympy import And, Eq, FiniteSet, Mul, Symbol
from sympy.core.relational import Relational

logger = logging.getLogger(__name__)

_BOXED_START = r"\boxed{"
# Parse only the box, with trailing units kept: `5 cm` reads as 5*c*m, not 5. Unlike
# Math-Verify's default, no fallback to other text or to a plain number in the box.
_KEEP_UNITS = [
    LatexExtractionConfig(
        try_extract_without_anchor=False,
        normalization_config=LatexNormalizationConfig(
            basic_latex=True, units=False, malformed_operators=True, nits=True
        ),
    )
]
# The thin space in `10\,002`; Math-Verify reads it as 10 + 002 = 12.
_THIN_SPACE_IN_NUMBER = re.compile(r"(?<=\d)\\,(?=\d{3}(?!\d))")
# `\leqq` and `\geqq`; Math-Verify reads `p\leqq0` as `p \le q \cdot 0`.
_LEQQ = re.compile(r"\\([lg]e)qq")
# A delimiter size like the `\bigl` in `\bigl(2-\sqrt3\bigr)` (not `\bigcup`), which Math-Verify cannot parse.
_DELIMITER_SIZE = re.compile(r"(?<!\\)\\[Bb]igg?[lrm]?(?![A-Za-z])")
# Where a trailing qualifier starts, once LaTeX spaces are plain spaces: `\text{ for all } n`,
# `\forall n`, `, k \in \mathbb{Z}`, `(a,b \in \mathbb{R})`.
_QUALIFIER = re.compile(
    r"\\text\{\s*for\s+(?:all|any|each|every|some)\b"
    r"|\\forall\b"
    r"|[,(]\s*[A-Za-z](?:\s*,\s*[A-Za-z]){0,3}\s*\\in\s*\\mathbb"
)
# A decimal gold like `16.67`, often rounded from an exact `\frac{50}{3}`.
_DECIMAL = re.compile(r"\s*-?(\d+)\.(\d+)\s*")
# A word answer like `indonesian`. 1-2 letters stay math: `ab`, or a segment `CD`.
_WORD = re.compile(r"\s*[A-Za-z]{3,}\s*")
# Spaces between letters, as in `D A E C B`, which would otherwise skip the word check.
_SPACING = re.compile(r"\s+|\\[,;:! ]")
# A LaTeX command or a non-letter, dropped to read the letters of an answer.
_NON_LETTER = re.compile(r"\\[A-Za-z]+|[^A-Za-z]")
# `60^\circ`, read as radians when only the other side has a `\pi`.
_DEGREES = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*\^\s*(?:\{\\circ\}|\\circ)")
# A `,` or `;` (not the spaces `\,` `\;`), or the word `or` or `and`: a list, not `name = value`.
_LIST = re.compile(r"(?<!\\)[,;]|\b(?:or|and)\b")
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
    A boxed `A_{\\min} = \\frac12` scores by its right side.
    Degrees read as radians when only one side has a `\\pi`: `60^\\circ` matches `\\frac{\\pi}{3}`.
    These answer and gold pairs also match, though Math-Verify alone misses them:
    - `p \\le 0` and `p \\leqq 0`;
    - `4\\pi\\bigl(2-\\sqrt3\\bigr)` and `4\\pi(2-\\sqrt3)`: a delimiter size;
    - `-12\\%` and `-12`: a `\\%` on one side;
    - `x = 2k\\pi,\\ k \\in \\mathbb{Z}` and `2k\\pi`: a trailing qualifier;
    - `-1 < x < 2` and `(-1, 2)`: a solved inequality and its interval;
    - `1` and `\\frac{9}{9} = 1`: a gold whose left side has no symbols;
    - `\\frac{50}{3}` and `16.67`: a gold rounded to 3+ significant digits;
    - `-c + 2` and `2 - c`: Math-Verify drops a trailing `c`, `m` or `h` as a unit.
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
    prediction = _LEQQ.sub(r"\\\1 ", prediction)
    ground_truth = _LEQQ.sub(r"\\\1 ", ground_truth)
    prediction = _DELIMITER_SIZE.sub("", prediction)
    ground_truth = _DELIMITER_SIZE.sub("", ground_truth)
    if "\\pi" in ground_truth and "\\pi" not in prediction:
        prediction = _DEGREES.sub(r"(\1\\pi/180)", prediction)
    elif "\\pi" in prediction and "\\pi" not in ground_truth:
        ground_truth = _DEGREES.sub(r"(\1\\pi/180)", ground_truth)
    boxed_text = prediction[len(_BOXED_START) : -1]

    # Compare words as strings: Math-Verify reads `tea` as t*e*a, so `eat` matches,
    # and reads the `I` in `Indonesian` as the imaginary unit.
    answer = _SPACING.sub("", boxed_text)
    if _WORD.fullmatch(answer) and _WORD.fullmatch(ground_truth):
        return float(answer.lower() == ground_truth.strip().lower())
    # Math-Verify reads a word gold as a product, so `(DAECB)` or `D\cdot A\cdot E\cdot C\cdot B`
    # would match `EDACB`: reject an answer whose letters reorder the gold's.
    if _WORD.fullmatch(ground_truth):
        letters = _NON_LETTER.sub("", boxed_text).lower()
        gold_letters = ground_truth.strip().lower()
        if letters != gold_letters and sorted(letters) == sorted(gold_letters):
            return 0.0

    try:
        if _matches(ground_truth, boxed_text):
            return 1.0
        # The retries below run only after a miss: run first, they break matches,
        # e.g. `12\%` would no longer match gold `0.12`.
        # Math-Verify matches `12\%` with `12`, but not `-12\%` with `-12`. Retry without `\%`,
        # so `0.5\%` also matches gold 0.5.
        if "\\%" in ground_truth + boxed_text and _matches(
            ground_truth.replace("\\%", ""), boxed_text.replace("\\%", "")
        ):
            return 1.0
        # Math-Verify reads `x = \frac{3\pi}{2} + 2\pi k,\ k \in \mathbb{Z}` as 2; retry without `, k ...`.
        unqualified = _drop_trailing_qualifier(boxed_text)
        if unqualified != boxed_text and score_math_response(
            _BOXED_START + unqualified + "}", ground_truth
        ):
            return 1.0
        # For an answer without `=`, Math-Verify drops a gold's left side only if it is all
        # symbols (`x = 5`). Drop one with no symbols too: `\frac{9}{9} = 1`, or `\gamma = 120^\circ`
        # (`\gamma` reads as Euler's constant). Skip an `=` in braces, as in `\sum_{k=1}^{10} k`.
        gold_left_side, _, gold_right_side = ground_truth.rpartition("=")
        if (
            gold_left_side
            and "=" not in boxed_text
            and gold_right_side.count("{") == gold_right_side.count("}")
        ):
            gold = parse(_BOXED_START + ground_truth + "}", parsing_timeout=None)
            if gold and isinstance(gold[0], Eq) and not gold[0].lhs.free_symbols:
                return score_math_response(response, gold_right_side)
        # Math-Verify reads `x = 5` as 5 but cannot parse a named left side like `A_{\min} = \frac12`.
        # Retry on the right side of the last `=`, unless that `=` is in `>=`, `<=`, `!=` or braces,
        # the box is a list, or the gold is an equation (`y = 3` would then match gold `x = 3`).
        left_side, _, right_side = boxed_text.rpartition("=")
        if (
            not left_side
            or "=" in ground_truth
            or left_side[-1] in "<>!"
            or right_side.count("{") != right_side.count("}")
            or _LIST.search(boxed_text)
        ):
            return 0.0
        return float(_matches(ground_truth, right_side))
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


def _matches(gold: str, answer: str) -> bool:
    """Return whether Math-Verify equates a gold and an answer, each boxed first.

    A decimal gold with 3+ significant digits compares at its own decimals (at most 6,
    Math-Verify's default), so `\\frac{50}{3}` matches `16.67`, while `\\frac{19}{100}` still
    misses `0.2`, which is likely exact. An exact `12.5` then also accepts `12.46`.
    """
    float_rounding = 6
    decimal = _DECIMAL.fullmatch(gold)
    if decimal and len((decimal[1] + decimal[2]).lstrip("0")) >= 3:
        float_rounding = min(len(decimal[2]), 6)
    # Box the gold like the answer: a bare `2\sqrt{3}` parses as 2, and a
    # bare `(1,2)` or `\pi/4` parses to nothing.
    gold_box, answer_box = _BOXED_START + gold + "}", _BOXED_START + answer + "}"
    gold_dropped = parse(gold_box, parsing_timeout=None)
    answer_dropped = parse(answer_box, parsing_timeout=None)
    gold_kept = parse(gold_box, extraction_config=_KEEP_UNITS, parsing_timeout=None)
    answer_kept = parse(answer_box, extraction_config=_KEEP_UNITS, parsing_timeout=None)
    answer_parsed = _drop_units(answer_dropped, answer_kept, other=gold_kept)
    gold_parsed = _drop_units(gold_dropped, gold_kept, other=answer_kept)
    return verify(
        gold_parsed,
        answer_parsed,
        float_rounding=float_rounding,
        # Also match a solved `-1 < x < 2` with gold `(-1, 2)`; the reverse matches by default.
        # Math-Verify would solve an unsolved `x^2 - 3x + 2 < 0`, e.g. copied from the problem.
        # Not against a finite set: Math-Verify equates an open interval with any 2-element set.
        allow_set_relation_comp=bool(answer_parsed)
        and _is_solved(answer_parsed[0])
        and not (gold_parsed and isinstance(gold_parsed[0], FiniteSet)),
        timeout_seconds=None,
    )


def _drop_units(dropped: list, kept: list, other: list) -> list:
    """Return `dropped`, Math-Verify's parse, or `kept` if a unit it drops is a variable.

    Math-Verify's units include `c`, `h` and `m`: it reads `2 - c` as `2 -`, `n = 4m` as
    `n = 4`, and `15+55m` as `15+55`. `dropped` is used if `kept` does not parse, or if:
    (a) `dropped` parses,
    (b) `other`, the other side with units kept, has none of the dropped symbols, and
    (c) the unit is in `\\text` (`2^{n-1} \\text{ ways}`) or multiplies a number (`5 cm`).
    """
    # `parse` gives [expression, normalized LaTeX], or [normalized LaTeX] if that does not parse.
    if not kept or isinstance(kept[0], str):
        return dropped
    if not dropped or isinstance(dropped[0], str):
        return kept
    dropped_symbols = kept[0].free_symbols - dropped[0].free_symbols
    other_symbols = (
        set() if not other or isinstance(other[0], str) else other[0].free_symbols
    )
    in_text = dropped[-1].count("\\text") < kept[-1].count("\\text")
    is_quantity = isinstance(kept[0], Mul) and not dropped[0].free_symbols
    if (in_text or is_quantity) and dropped_symbols.isdisjoint(other_symbols):
        return dropped
    return kept


def _is_solved(answer: object) -> bool:
    """Return whether each relation compares a lone symbol with a constant, as in `-1 < x < 2`."""
    relations = answer.args if isinstance(answer, And) else (answer,)
    return all(
        isinstance(relation, Relational)
        and (
            (isinstance(relation.lhs, Symbol) and not relation.rhs.free_symbols)
            or (isinstance(relation.rhs, Symbol) and not relation.lhs.free_symbols)
        )
        for relation in relations
    )


def _drop_trailing_qualifier(answer: str) -> str:
    """Drop a trailing qualifier: `x = 2k\\pi,\\ k \\in \\mathbb{Z}` becomes `x = 2k\\pi`.

    Keeps a piecewise answer whole: one with `\\begin{cases}` rows, or with a `=`, `,`, `;` or
    `\\text` after the qualifier, as in `2 \\forall n \\ge 3,\\ 1\\ (n < 3)`.
    """
    # A `\\quad` separates cases like a `,`.
    spaced = _SPACING.sub(" ", answer).replace("\\qquad", ",").replace("\\quad", ",")
    match = _QUALIFIER.search(spaced)
    if (
        match is None
        or re.search(r"&|\\\\", answer)
        or re.search(r"[=,;]|\\text", spaced[match.end() :])
        or _QUALIFIER.search(spaced, match.end())
    ):
        return answer
    return spaced[: match.start()].rstrip(" ,")


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
