# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from torchtitan.tools.logging import logger

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list


_TEST_SUITES_FUNCTION = {
    "features": build_features_test_list,
    "models": build_model_tests_list,
    "h100": build_h100_tests_list,
}


class GPUPool:
    """Allocator for a fixed-size pool of physical GPU ids.

    ``acquire(n)`` blocks until ``n`` GPUs are free and returns a sorted list
    of ids; ``release`` returns them to the pool.
    """

    def __init__(self, total: int):
        self._free: list[int] = list(range(total))
        self._cond = threading.Condition()
        self.total = total

    def acquire(self, n: int) -> list[int]:
        with self._cond:
            while len(self._free) < n:
                self._cond.wait()
            chosen = sorted(self._free[:n])
            self._free = self._free[n:]
            return chosen

    def release(self, gpus: list[int]) -> None:
        with self._cond:
            self._free.extend(gpus)
            self._cond.notify_all()


_OUTPUT_LOCK = threading.Lock()


def _run_cmd_capture(cmd: str, prefix: str, timeout: float | None = None):
    """Run ``cmd``, capturing merged stdout/stderr into memory.

    Output is *not* streamed to the parent in real time: when running tests
    concurrently we want each test's log to appear as one contiguous block
    rather than interleaved line-by-line with other tests.

    A copy of the live output is written to ``{output_dir}``-rooted log files
    by the caller if desired; this function just buffers everything and
    returns it.

    Returns ``(returncode, captured_text)``. Raises
    ``subprocess.TimeoutExpired`` if ``timeout`` is exceeded.
    """
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    timed_out: list[bool] = []
    timer: threading.Timer | None = None
    if timeout is not None:

        def _kill() -> None:
            timed_out.append(True)
            try:
                proc.kill()
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout, _kill)
        timer.daemon = True
        timer.start()

    captured: list[str] = []
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            captured.append(raw.rstrip("\n"))
    finally:
        proc.wait()
        if timer is not None:
            timer.cancel()

    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)

    return proc.returncode, "\n".join(captured)


def _emit_block(prefix: str, header: str, body: str, footer: str = "") -> None:
    """Atomically write a multi-line block prefixed with ``[prefix] ``.

    Holds the global output lock for the entire block so concurrent tests do
    not interleave their lines.
    """
    with _OUTPUT_LOCK:
        sys.stderr.write(header)
        for line in body.splitlines():
            sys.stderr.write(f"[{prefix}] {line}\n")
        if footer:
            sys.stderr.write(footer)
        sys.stderr.flush()


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    module: str | None = None,
    config: str | None = None,
    gpu_ids: list[int] | None = None,
):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--dump_folder {output_dir}/{test_name}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    # When running in parallel, pin each test to a disjoint subset of physical
    # GPUs. Setting both CUDA_/HIP_VISIBLE_DEVICES makes this a no-op for the
    # arch that doesn't apply.
    if gpu_ids is not None:
        visible = ",".join(map(str, gpu_ids))
        gpu_env_prefix = (
            f"CUDA_VISIBLE_DEVICES={visible} HIP_VISIBLE_DEVICES={visible} "
        )
    else:
        gpu_env_prefix = ""

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = ""
        if module is not None:
            cmd += f"MODULE={module} "
        if config is not None:
            cmd += f"CONFIG={config} "
        cmd = (
            f"{gpu_env_prefix}NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
            f"./run_train.sh"
        )

        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd

        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)

        # save checkpoint (idx == 0) and load it for generation (idx == 1)
        if test_name == "test_generate" and idx == 1:
            cmd = (
                f"{gpu_env_prefix}NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
                f"CHECKPOINT_DIR={output_dir}/{test_name}/checkpoint/step-10 "
                "PROMPT='What is the meaning of life?' "
                f"./scripts/generate/run_llama_generate.sh --out > {output_dir}/{test_name}/generated_output.json"
            )

        start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            returncode, captured = _run_cmd_capture(
                cmd, prefix=test_name, timeout=test_flavor.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"\nTest timed out after {test_flavor.timeout}s: {test_flavor.test_descr}.\n"
                f"Command: {cmd}\n"
            ) from e

        end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"===== [{test_name}] start {start_ts} end {end_ts} "
            f"flavor: {test_flavor.test_descr} (rc={returncode}) =====\n"
            f"===== [{test_name}] command: {cmd} =====\n"
        )
        footer = f"===== [{test_name}] end of output (rc={returncode}) =====\n"
        _emit_block(test_name, header, captured, footer)

        if returncode != 0:
            tail = "\n".join(captured.splitlines()[-50:])
            raise RuntimeError(
                f"\nFailed test flavor: {test_flavor.test_descr} (rc={returncode}).\n"
                f"Command: {cmd}\n"
                f"Last 50 lines:\n{tail}\n"
            )


def _filter_tests(
    args, test_list: list[OverrideDefinitions]
) -> tuple[list[OverrideDefinitions], list[OverrideDefinitions]]:
    """Filter tests by --test_name / --exclude / disabled / arch / ngpu.

    Returns (runnable, skipped_due_to_ngpu).
    """
    exclude_set = set()
    if hasattr(args, "exclude") and args.exclude:
        exclude_set = {name.strip() for name in args.exclude.split(",")}

    runnable: list[OverrideDefinitions] = []
    skipped_ngpu: list[OverrideDefinitions] = []
    for test_flavor in test_list:
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue
        if test_flavor.disabled or test_flavor.test_name in exclude_set:
            continue
        if (
            getattr(args, "gpu_arch_type", "cuda") == "rocm"
            and test_flavor.skip_rocm_test
        ):
            continue
        if args.ngpu < test_flavor.ngpu:
            skipped_ngpu.append(test_flavor)
            continue
        runnable.append(test_flavor)
    return runnable, skipped_ngpu


def run_tests(args, test_list: list[OverrideDefinitions], module=None, config=None):
    """Run all integration tests to test the core features of TorchTitan."""
    runnable, skipped_ngpu = _filter_tests(args, test_list)
    for test_flavor in skipped_ngpu:
        logger.info(
            f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
            f" because --ngpu arg is {args.ngpu}"
        )

    failed_tests: list[tuple[str, str]] = []

    if args.parallel and runnable:
        # Schedule tests concurrently, packing them onto a fixed pool of
        # physical GPUs. A test can run as soon as `test_flavor.ngpu` GPUs are
        # free; the sum of in-flight test ngpu never exceeds `args.ngpu`.
        pool = GPUPool(args.ngpu)
        # Sort largest-first to reduce fragmentation / head-of-line blocking.
        scheduled = sorted(runnable, key=lambda t: -t.ngpu)
        # Worst case: every test wants 1 GPU and runs in parallel.
        max_workers = max(1, min(len(scheduled), args.ngpu))

        def _runner(test_flavor: OverrideDefinitions) -> None:
            gpus = pool.acquire(test_flavor.ngpu)
            logger.info(
                f"[parallel] {test_flavor.test_name}: acquired GPUs {gpus} "
                f"(ngpu={test_flavor.ngpu})"
            )
            try:
                run_single_test(
                    test_flavor, args.output_dir, module, config, gpu_ids=gpus
                )
            finally:
                pool.release(gpus)
                logger.info(f"[parallel] {test_flavor.test_name}: released GPUs {gpus}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures: dict[Future, OverrideDefinitions] = {
                ex.submit(_runner, t): t for t in scheduled
            }
            for fut in futures:
                test_flavor = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.error(str(e))
                    failed_tests.append((test_flavor.test_name, str(e)))
    else:
        for test_flavor in runnable:
            try:
                run_single_test(test_flavor, args.output_dir, module, config)
            except Exception as e:
                logger.error(str(e))
                failed_tests.append((test_flavor.test_name, str(e)))

    ran_any_test = bool(runnable)

    if failed_tests:
        failure_summary = "\n".join(
            f"  {name}: {error}" for name, error in failed_tests
        )
        raise RuntimeError(
            f"{len(failed_tests)} integration test(s) failed:\n{failure_summary}"
        )

    if not ran_any_test:
        available_tests = [t.test_name for t in test_list if not t.disabled]
        if hasattr(args, "test_suite"):
            logger.warning(
                f"No tests were run for --test_name '{args.test_name}' in test suite '{args.test_suite}'.\n"
                f"Available test names in '{args.test_suite}' suite: {available_tests}"
            )
        else:
            logger.warning(
                f"No tests were run for --test_name '{args.test_name}'.\n"
                f"Available test names: {available_tests}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", help="Directory to dump results generated by tests"
    )
    parser.add_argument(
        "--gpu_arch_type",
        default="cuda",
        choices=["cuda", "rocm"],
        help="GPU architecture type. Must be specified as either 'cuda' or 'rocm'.",
    )
    parser.add_argument(
        "--test_suite",
        default="features",
        choices=["features", "models", "h100"],
        help="Which test suite to run. If not specified, torchtitan composability tests will be run",
    )
    parser.add_argument(
        "--module",
        default="llama3",
        help="Model module to use for training (default: llama3). "
        "This is passed as MODULE env var to run_train.sh.",
    )
    parser.add_argument(
        "--config",
        default="llama3_debugmodel",
        help="Config function to use for training (default: llama3_debugmodel). "
        "This is passed as CONFIG env var to run_train.sh.",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test name to run (e.g., 'tp_only', 'full_checkpoint'). Use 'all' to run all tests (default: all)",
    )
    parser.add_argument(
        "--ngpu", default=8, type=int, help="Maximum number of GPUs to use"
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated list of test names to skip",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run tests concurrently, packing them onto the GPU pool. "
        "At most --ngpu GPUs are in use at any time; each test is pinned to a "
        "disjoint subset via CUDA_/HIP_VISIBLE_DEVICES. "
        "Use --no-parallel to force sequential execution (default: parallel).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    assert (
        args.test_suite in _TEST_SUITES_FUNCTION
    ), f"Unknown test suite {args.test_suite}"

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
