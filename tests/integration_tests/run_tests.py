# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from tests.integration_tests import OverrideDefinitions, validate_fake_pg_compatibility
from tests.integration_tests.b200 import build_b200_tests_list
from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list


_TEST_SUITES_FUNCTION = {
    "features": build_features_test_list,
    "models": build_model_tests_list,
    "h100": build_h100_tests_list,
    "b200": build_b200_tests_list,
}

# Held while a test writes its captured output so concurrent tests do not
# interleave their lines.
_OUTPUT_LOCK = threading.Lock()


def _parse_test_suites(value: str) -> tuple[str, ...]:
    suites = tuple(part.strip() for part in value.split(",") if part.strip())
    if not suites:
        raise ValueError("--test_suite must contain at least one suite")
    unknown = tuple(suite for suite in suites if suite not in _TEST_SUITES_FUNCTION)
    if unknown:
        available = ", ".join(_TEST_SUITES_FUNCTION)
        raise ValueError(
            f"Unknown test suite(s): {', '.join(unknown)}. Available: {available}"
        )
    if len(set(suites)) != len(suites):
        raise ValueError("--test_suite must not contain duplicate suites")
    return suites


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


def _run_cmd(
    cmd: str,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` in a shell, capturing merged stdout/stderr into memory.

    Output is *not* streamed to the parent in real time: when running tests
    concurrently we want each test's log to appear as one contiguous block
    rather than interleaved line-by-line with other tests.

    On timeout, returns a synthetic ``CompletedProcess`` with ``returncode=-1``
    and ``stdout`` populated with whatever the child had emitted so far, so
    callers do not need to special-case ``TimeoutExpired``.
    """
    try:
        return subprocess.run(
            [cmd],
            encoding="utf-8",
            errors="replace",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        stdout = (
            e.stdout.decode("utf-8", errors="replace")
            if isinstance(e.stdout, bytes)
            else (e.stdout or "")
        )
        return subprocess.CompletedProcess(cmd, -1, stdout=stdout, stderr=None)


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


def _join_override_args(override_args: tuple[str, ...]) -> str:
    """Safely join legacy shell fragments into a command line."""
    return shlex.join(
        token for fragment in override_args for token in shlex.split(fragment)
    )


def _read_golden_spec(golden_numerics_path: Path) -> tuple[int, tuple[str, ...]]:
    columns = ("step", "loss")
    steps: list[int] = []
    with golden_numerics_path.open() as golden_file:
        for line in golden_file:
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == "#" and len(fields) > 1 and fields[1] == "step":
                columns = tuple(fields[1:])
            elif fields[0] != "#":
                steps.append(int(fields[0]))
    if not steps:
        raise ValueError(f"Numerics golden has no steps: {golden_numerics_path}")
    return max(steps), columns[1:]


def _parallelism_summary(config: Trainer.Config, world_size: int) -> str:
    parallelism = config.parallelism
    fsdp_degree = parallelism.data_parallel_shard_degree
    if fsdp_degree == -1:
        dense_parallel_degree = (
            parallelism.data_parallel_replicate_degree
            * parallelism.context_parallel_degree
            * parallelism.tensor_parallel_degree
            * parallelism.pipeline_parallel_degree
        )
        fsdp_degree = world_size // dense_parallel_degree
    dimensions = (
        ("FSDP", fsdp_degree),
        ("TP", parallelism.tensor_parallel_degree),
        ("CP", parallelism.context_parallel_degree),
        ("EP", parallelism.expert_parallel_degree),
        ("PP", parallelism.pipeline_parallel_degree),
    )
    return ", ".join(
        f"{name}={degree}"
        for name, degree in dimensions
        if name == "FSDP" or degree > 1
    )


def _add_parallelism_header(result_path: Path, parallelism: str) -> None:
    lines = result_path.read_text().splitlines()
    insert_at = next(
        (index + 1 for index, line in enumerate(lines) if line.startswith("# ngpu:")),
        0,
    )
    lines.insert(insert_at, f"# parallelism: {parallelism}")
    result_path.write_text("\n".join(lines) + "\n")


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    *,
    use_fake_pg: bool = False,
    export_numerics: bool = False,
    # ``gpu_ids`` is set only in parallel mode; sequential runs leave the
    # child process to use all visible GPUs.
    gpu_ids: list[int] | None = None,
):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--dump_folder {output_dir}/{test_name}"

    if test_flavor.golden_numerics_path is not None and len(test_flavor.configs) != 1:
        raise ValueError(
            f"{test_name} sets golden_numerics_path but defines "
            f"{len(test_flavor.configs)} configs; numerics tests must define "
            "exactly one config"
        )

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    # When running in parallel, pin each test to a disjoint subset of physical
    # GPUs. Setting both CUDA_/HIP_VISIBLE_DEVICES makes this a no-op for the
    # architecture that does not apply.
    base_env = os.environ.copy()
    if gpu_ids is not None:
        visible = ",".join(map(str, gpu_ids))
        base_env["CUDA_VISIBLE_DEVICES"] = visible
        base_env["HIP_VISIBLE_DEVICES"] = visible
    base_env["NGPU"] = str(test_flavor.ngpu)
    base_env["LOG_RANK"] = all_ranks
    base_env.pop("COMM_MODE", None)
    if use_fake_pg:
        base_env["COMM_MODE"] = "fake_backend"

    for run, override_arg in enumerate(test_flavor.override_args):
        test_output_dir = str(Path(output_dir) / test_name)
        config_fn = test_flavor.configs[run] if test_flavor.configs else None
        config = config_fn() if config_fn is not None else None
        if use_fake_pg and config is not None:
            validate_fake_pg_compatibility(test_flavor, config)
        env = base_env.copy()
        env["TORCHTITAN_TEST_OUTPUT_DIR"] = test_output_dir
        if config_fn is not None:
            env["MODULE"] = config_fn.__module__
            env["CONFIG"] = config_fn.__name__
        override_arg = tuple(
            arg.replace("{test_output_dir}", test_output_dir) for arg in override_arg
        )
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if test_flavor.golden_numerics_path is not None:
            # Reuse this integration run for numerics: loss_compare.py runs the
            # config once, extracts full-precision TensorBoard metrics, and
            # compares them with the mode-specific golden (or exports them).
            assert config_fn is not None and config is not None
            execution_mode = "fake_pg" if use_fake_pg else "real_pg"
            golden_numerics_path = Path(
                test_flavor.golden_numerics_path.format(execution_mode=execution_mode)
            )
            if export_numerics:
                steps = config.training.steps
                metrics = ("loss", "grad_norm")
                result_path = Path(output_dir) / golden_numerics_path.name
                result_arg = f"--export-result={result_path}"
            else:
                steps, metrics = _read_golden_spec(golden_numerics_path)
                result_path = golden_numerics_path
                result_arg = f"--import-result={golden_numerics_path}"

            options = _join_override_args(override_arg)
            command = [
                sys.executable,
                "scripts/loss_compare.py",
                ".",
                ".",
                f"--baseline-module={config_fn.__module__}",
                f"--baseline-config={config_fn.__name__}",
                f"--baseline-options={options}",
                f"--test-module={config_fn.__module__}",
                f"--test-config={config_fn.__name__}",
                f"--test-options={options}",
                f"--job-dump-folder={Path(output_dir) / test_name}",
                f"--metrics={','.join(metrics)}",
                f"--steps={steps}",
                f"--baseline-ngpus={test_flavor.ngpu}",
                f"--test-ngpus={test_flavor.ngpu}",
                result_arg,
            ]
            if not export_numerics:
                command.append("--assert-equal")
            if use_fake_pg:
                command.append("--no-seed-checkpoint")

            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=test_flavor.timeout,
            )
            cmd = shlex.join(command)
            if export_numerics and result.returncode == 0:
                _add_parallelism_header(
                    result_path,
                    _parallelism_summary(config, test_flavor.ngpu),
                )
        else:
            # Tests without a golden run directly and guard E2E execution only;
            # they do not assert loss or gradient-norm values.
            env["TORCH_TRACE"] = f"{output_dir}/{test_name}/compile_trace"
            cmd = f"./run_train.sh {dump_folder_arg}"
            if override_arg:
                cmd += " " + _join_override_args(override_arg)
            result = _run_cmd(cmd, timeout=test_flavor.timeout, env=env)
        returncode = result.returncode
        captured = result.stdout or ""

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
            # ``_run_cmd`` returns rc=-1 to signal a timeout.
            reason = (
                f"timed out after {test_flavor.timeout}s"
                if returncode == -1
                else f"rc={returncode}"
            )
            raise RuntimeError(
                f"\nFailed test flavor: {test_flavor.test_descr} ({reason}).\n"
                f"Command: {cmd}\n"
                f"Last 50 lines:\n{tail}\n"
            )


def _filter_tests(
    args, test_list: list[OverrideDefinitions]
) -> tuple[list[OverrideDefinitions], list[OverrideDefinitions]]:
    """Filter tests by name, scope, disabled state, architecture, and GPU count.

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
        execution_mode = getattr(args, "execution_mode", "real_pg")
        if execution_mode == "fake_pg" and test_flavor.use_real_pg:
            continue
        if (
            getattr(args, "test_scope", "all") == "real_pg_required"
            and not test_flavor.use_real_pg
        ):
            continue
        if test_flavor.disabled or test_flavor.test_name in exclude_set:
            continue
        if (
            getattr(args, "gpu_arch_type", "cuda") == "rocm"
            and test_flavor.skip_rocm_test
        ):
            continue
        if execution_mode != "fake_pg" and args.ngpu < test_flavor.ngpu:
            skipped_ngpu.append(test_flavor)
            continue
        runnable.append(test_flavor)
    return runnable, skipped_ngpu


def run_tests(
    args,
    test_list: list[OverrideDefinitions],
    parallel: bool = True,
):
    """Run all integration tests to test the core features of TorchTitan."""
    runnable, skipped_ngpu = _filter_tests(args, test_list)
    for test_flavor in skipped_ngpu:
        logger.info(
            f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
            f" because --ngpu arg is {args.ngpu}"
        )
    failed_tests: list[tuple[str, str]] = []
    execution_mode = getattr(args, "execution_mode", "real_pg")
    export_numerics = getattr(args, "export_numerics", False)

    def physical_ngpu(test_flavor: OverrideDefinitions) -> int:
        return 1 if execution_mode == "fake_pg" else test_flavor.ngpu

    if parallel and runnable:
        # Schedule tests concurrently, packing them onto a fixed pool of
        # physical GPUs. Fake PG tests consume one physical GPU while retaining
        # test_flavor.ngpu as the simulated world size.
        pool = GPUPool(args.ngpu)
        # Submit largest-first so the very first wave packs efficiently and
        # avoids head-of-line blocking by an oversized test arriving late.
        # NOTE: this only deterministically orders the *first* batch; once
        # workers start finishing at different times, subsequent acquisition
        # order is driven by completion times, not by ``ngpu``.
        scheduled = sorted(runnable, key=lambda t: -physical_ngpu(t))
        # Worst case: every test wants 1 GPU and runs in parallel.
        max_workers = max(1, min(len(scheduled), args.ngpu))

        def _runner(test_flavor: OverrideDefinitions) -> None:
            gpus = pool.acquire(physical_ngpu(test_flavor))
            logger.info(
                f"[parallel] {test_flavor.test_name}: acquired GPUs {gpus} "
                f"(ngpu={test_flavor.ngpu})"
            )
            try:
                run_single_test(
                    test_flavor,
                    args.output_dir,
                    use_fake_pg=execution_mode == "fake_pg",
                    export_numerics=export_numerics,
                    gpu_ids=gpus,
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
                run_single_test(
                    test_flavor,
                    args.output_dir,
                    use_fake_pg=execution_mode == "fake_pg",
                    export_numerics=export_numerics,
                )
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
        help="Comma-separated test suites to run: features, models, h100, b200.",
    )
    parser.add_argument(
        "--execution_mode",
        choices=("fake_pg", "real_pg"),
        default="real_pg",
        help="Communication mode used to execute the selected suites.",
    )
    parser.add_argument(
        "--test_scope",
        choices=("all", "real_pg_required"),
        default="all",
        help="Run every selected test or only tests marked use_real_pg=True.",
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
        "--export-numerics",
        action="store_true",
        help=(
            "Export results for tests with golden_numerics_path instead of "
            "comparing."
        ),
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

    try:
        test_suites = _parse_test_suites(args.test_suite)
    except ValueError as error:
        parser.error(str(error))
    hardware_suites = {"h100", "b200"}.intersection(test_suites)
    if args.execution_mode == "fake_pg" and hardware_suites:
        suites = ", ".join(sorted(hardware_suites))
        parser.error(f"The {suites} suite(s) only support --execution_mode real_pg")
    if args.execution_mode == "fake_pg" and args.test_scope == "real_pg_required":
        parser.error("real_pg_required test scope requires --execution_mode real_pg")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    for test_suite in test_suites:
        suite_args = argparse.Namespace(**vars(args))
        suite_args.test_suite = test_suite
        if len(test_suites) > 1:
            suite_args.output_dir = os.path.join(args.output_dir, test_suite)
            os.makedirs(suite_args.output_dir)

        test_list = _TEST_SUITES_FUNCTION[test_suite]()
        run_tests(suite_args, test_list, parallel=suite_args.parallel)


if __name__ == "__main__":
    main()
