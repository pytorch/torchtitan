# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-isolated SM100 tuning for the DSV3 CODA FlexGEMM benchmarks.

QuACK configurations are compiled in separate processes because a broken
candidate can fail or hang inside a CUDA kernel. Multi-GEMM patterns use
coordinate descent: one FlexGEMM configuration is swept while the other
winning configurations remain fixed.

Select the graph-grounded model shape inventory with ``--suite 671b`` or
``--suite 16b``.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__:
    from torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench import (
        CASES as CASES_671B,
    )
    from torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench_16b import (
        CASES as CASES_16B,
    )
else:
    from coda_fusion_microbench import (  # pyrefly: ignore [missing-import]
        CASES as CASES_671B,
    )
    from coda_fusion_microbench_16b import (  # pyrefly: ignore [missing-import]
        CASES as CASES_16B,
    )


CASES = CASES_671B
SUITES = {"671b": CASES_671B, "16b": CASES_16B}


Config = dict[str, Any]


def _microbenchmark_command(suite: str = "671b") -> list[str]:
    module_suffix = (
        "coda_fusion_microbench_16b" if suite == "16b" else "coda_fusion_microbench"
    )
    if __package__:
        return [
            sys.executable,
            "-m",
            f"torchtitan.experiments.graph_trainer.benchmarks.{module_suffix}",
        ]
    return [sys.executable, str(Path(__file__).with_name(f"{module_suffix}.py"))]


def _config(
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    is_dynamic_persistent: bool,
) -> Config:
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "is_dynamic_persistent": is_dynamic_persistent,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "swap_ab": False,
        "use_tma_gather": False,
    }


PRIORITY_CONFIGS = (
    _config(128, 256, 2, 1, True),
    _config(128, 192, 2, 1, True),
    _config(256, 256, 2, 1, True),
    _config(256, 256, 2, 2, True),
    _config(256, 192, 2, 1, True),
    _config(128, 128, 1, 1, False),
    _config(128, 256, 1, 1, True),
    _config(128, 256, 1, 1, False),
    _config(128, 128, 2, 1, True),
    _config(256, 128, 2, 1, True),
    _config(128, 224, 1, 1, True),
    _config(128, 160, 1, 1, True),
)

WIDE_REDUCTION_CONFIG = _config(256, 512, 2, 1, False)
LARGE_GEMM_CONFIG = _config(256, 256, 2, 2, True)

INITIAL_CONFIGS: dict[str, tuple[Config, ...]] = {
    "b1_lm_head_input_grad_cast": (LARGE_GEMM_CONFIG,),
    "b2_shared_expert_swiglu_backward": (
        LARGE_GEMM_CONFIG,
        LARGE_GEMM_CONFIG,
    ),
    "b4_router_input_grad_add": (PRIORITY_CONFIGS[1],),
    "b5_mla_q_rmsnorm_backward": (PRIORITY_CONFIGS[0],),
    "b6_weight_grad_cast": (LARGE_GEMM_CONFIG,),
    "b7_attention_input_grad_merge": (LARGE_GEMM_CONFIG,),
    "f2_q_rmsnorm": (WIDE_REDUCTION_CONFIG, LARGE_GEMM_CONFIG),
    "f2_kv_rmsnorm": (PRIORITY_CONFIGS[0], LARGE_GEMM_CONFIG),
    "f3_attention_output": (WIDE_REDUCTION_CONFIG,),
    "f3_moe_output": (WIDE_REDUCTION_CONFIG,),
    "f4_shared_expert_swiglu": (LARGE_GEMM_CONFIG, LARGE_GEMM_CONFIG),
    "f6_router_sigmoid_bias": (PRIORITY_CONFIGS[1],),
}

INITIAL_CONFIGS_16B: dict[str, tuple[Config, ...]] = {
    "b1_lm_head_input_grad_cast": (LARGE_GEMM_CONFIG,),
    "b2_shared_expert_swiglu_backward": (
        LARGE_GEMM_CONFIG,
        LARGE_GEMM_CONFIG,
    ),
    "b2_dense_ffn_swiglu_backward": (
        LARGE_GEMM_CONFIG,
        LARGE_GEMM_CONFIG,
    ),
    "b4_router_input_grad_add": (PRIORITY_CONFIGS[1],),
    "b5_mla_kv_rmsnorm_backward": (WIDE_REDUCTION_CONFIG,),
    "b6_shared_expert_weight_grad_cast": (LARGE_GEMM_CONFIG,),
    "b7_attention_input_grad_merge": (LARGE_GEMM_CONFIG,),
    "f2_kv_rmsnorm": (PRIORITY_CONFIGS[0], LARGE_GEMM_CONFIG),
    "f3_attention_output": (WIDE_REDUCTION_CONFIG,),
    "f3_moe_output": (WIDE_REDUCTION_CONFIG,),
    "f3_dense_ffn_output": (WIDE_REDUCTION_CONFIG,),
    "f4_shared_expert_swiglu": (LARGE_GEMM_CONFIG, LARGE_GEMM_CONFIG),
    "f4_dense_ffn_swiglu": (LARGE_GEMM_CONFIG, LARGE_GEMM_CONFIG),
}


@dataclasses.dataclass(frozen=True)
class CandidateResult:
    index: int
    configs: tuple[Config, ...]
    status: str
    device: int
    median_ms: float | None
    compiled_eager_ms: float | None
    flex_to_compiled_eager: float | None
    result_path: str
    log_path: str
    error: str | None = None


def _full_sm100_configs() -> tuple[Config, ...]:
    tile_cluster_shapes = (
        *(
            (128, tile_n, cluster_m, cluster_n)
            for tile_n in (64, 128, 160, 192, 224, 256)
            for cluster_m, cluster_n in ((1, 1), (1, 2), (2, 1), (2, 2))
        ),
        *(
            (256, tile_n, cluster_m, cluster_n)
            for tile_n in (64, 128, 160, 192, 224, 256)
            for cluster_m, cluster_n in ((2, 1), (2, 2))
        ),
        (256, 512, 2, 1),
    )
    configs = [
        _config(tile_m, tile_n, cluster_m, cluster_n, is_dynamic_persistent)
        for tile_m, tile_n, cluster_m, cluster_n in tile_cluster_shapes
        for is_dynamic_persistent in (True, False)
    ]
    priority_keys = {_config_key(config) for config in PRIORITY_CONFIGS}
    configs.sort(
        key=lambda config: (
            0 if _config_key(config) in priority_keys else 1,
            config["tile_m"],
            config["tile_n"],
            config["cluster_m"],
            config["cluster_n"],
            not config["is_dynamic_persistent"],
        )
    )
    return tuple(configs)


def _config_key(config: Config) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _deduplicate_configs(configs: Sequence[Config]) -> tuple[Config, ...]:
    unique = {}
    for config in configs:
        unique.setdefault(_config_key(config), config)
    return tuple(unique.values())


def _deduplicate_config_sets(
    config_sets: Sequence[tuple[Config, ...]],
) -> tuple[tuple[Config, ...], ...]:
    unique = {}
    for configs in config_sets:
        key = json.dumps(configs, sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, configs)
    return tuple(unique.values())


def _search_configs(search: str, max_candidates: int | None) -> tuple[Config, ...]:
    configs = (
        PRIORITY_CONFIGS
        if search == "priority"
        else _deduplicate_configs((*PRIORITY_CONFIGS, *_full_sm100_configs()))
    )
    return configs if max_candidates is None else configs[:max_candidates]


def _parse_config(value: str) -> Config:
    config = json.loads(value)
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("configuration must be a JSON object")
    return config


def _initial_configs(args: argparse.Namespace) -> tuple[Config, ...]:
    suite = getattr(args, "suite", "671b")
    cases = SUITES[suite]
    initial_configs = INITIAL_CONFIGS_16B if suite == "16b" else INITIAL_CONFIGS
    case = cases[args.case]
    if args.base_config:
        configs = tuple(args.base_config)
        if len(configs) == 1:
            configs *= case.num_flex_gemms
    else:
        configs = initial_configs[args.case]
    if len(configs) != case.num_flex_gemms:
        raise ValueError(
            f"{args.case} requires {case.num_flex_gemms} base configurations, "
            f"got {len(configs)}"
        )
    return configs


def _result_metrics(path: Path) -> tuple[float, float, float]:
    with path.open(encoding="utf-8") as result_file:
        results = json.load(result_file)
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected one benchmark result in {path}")
    result = results[0]
    if not all(report["passed"] for report in result["correctness"]["flex_gemm"]):
        raise ValueError(f"correctness failed in {path}")
    flex_ms = float(result["flex_gemm"]["median_ms"])
    compiled_eager_ms = float(result["compiled_eager"]["median_ms"])
    return flex_ms, compiled_eager_ms, flex_ms / compiled_eager_ms


def _candidate_name(index: int, configs: Sequence[Config]) -> str:
    digest = hashlib.sha1(
        json.dumps(configs, sort_keys=True).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:10]
    return f"candidate_{index:03d}_{digest}"


def _run_candidate(
    *,
    index: int,
    configs: tuple[Config, ...],
    device_queue: queue.Queue[int],
    output_dir: Path,
    args: argparse.Namespace,
) -> CandidateResult:
    device = device_queue.get()
    candidate_name = _candidate_name(index, configs)
    result_path = output_dir / f"{candidate_name}.json"
    log_path = output_dir / f"{candidate_name}.log"
    metadata_path = output_dir / f"{candidate_name}.metadata.json"
    cache_path = output_dir / f".{candidate_name}_cache"
    try:
        if args.resume and result_path.exists() and metadata_path.exists():
            try:
                with metadata_path.open(encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                median_ms, compiled_eager_ms, normalized_ratio = _result_metrics(
                    result_path
                )
                return CandidateResult(
                    index=index,
                    configs=configs,
                    status="passed",
                    device=int(metadata["physical_device"]),
                    median_ms=median_ms,
                    compiled_eager_ms=compiled_eager_ms,
                    flex_to_compiled_eager=normalized_ratio,
                    result_path=str(result_path),
                    log_path=str(log_path),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                result_path.unlink()

        command = [
            *_microbenchmark_command(args.suite),
            "--case",
            args.case,
            "--device",
            "0",
            "--warmup",
            str(args.warmup),
            "--rounds",
            str(args.rounds),
            "--iterations",
            str(args.iterations),
            "--output",
            str(result_path),
        ]
        for config in configs:
            command.extend(("--config", json.dumps(config, separators=(",", ":"))))

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(device),
                "TORCH_NATIVE_SKIP_VERSION_CHECK": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(cache_path),
            }
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
            )
            try:
                return_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return CandidateResult(
                    index=index,
                    configs=configs,
                    status="timeout",
                    device=device,
                    median_ms=None,
                    compiled_eager_ms=None,
                    flex_to_compiled_eager=None,
                    result_path=str(result_path),
                    log_path=str(log_path),
                    error=f"exceeded {args.timeout}s",
                )
        if return_code != 0:
            return CandidateResult(
                index=index,
                configs=configs,
                status="failed",
                device=device,
                median_ms=None,
                compiled_eager_ms=None,
                flex_to_compiled_eager=None,
                result_path=str(result_path),
                log_path=str(log_path),
                error=f"exit code {return_code}",
            )
        median_ms, compiled_eager_ms, normalized_ratio = _result_metrics(result_path)
        _write_json(
            metadata_path,
            {
                "physical_device": device,
                "configs": configs,
                "median_ms": median_ms,
                "compiled_eager_ms": compiled_eager_ms,
                "flex_to_compiled_eager": normalized_ratio,
            },
        )
        return CandidateResult(
            index=index,
            configs=configs,
            status="passed",
            device=device,
            median_ms=median_ms,
            compiled_eager_ms=compiled_eager_ms,
            flex_to_compiled_eager=normalized_ratio,
            result_path=str(result_path),
            log_path=str(log_path),
        )
    except Exception as error:
        return CandidateResult(
            index=index,
            configs=configs,
            status="failed",
            device=device,
            median_ms=None,
            compiled_eager_ms=None,
            flex_to_compiled_eager=None,
            result_path=str(result_path),
            log_path=str(log_path),
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if not args.keep_cache:
            shutil.rmtree(cache_path, ignore_errors=True)
        device_queue.put(device)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.write("\n")


def _tune_flex_gemm(
    *,
    flex_index: int,
    pass_index: int,
    current_configs: tuple[Config, ...],
    candidates: tuple[Config, ...],
    device_queue: queue.Queue[int],
    root_output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Config, ...]:
    output_dir = root_output_dir / f"pass_{pass_index}" / f"flex_{flex_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _deduplicate_configs((current_configs[flex_index], *candidates))
    jobs = []
    for candidate in candidates:
        configs = list(current_configs)
        configs[flex_index] = candidate
        jobs.append(tuple(configs))

    results = []
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = {
            executor.submit(
                _run_candidate,
                index=index,
                configs=configs,
                device_queue=device_queue,
                output_dir=output_dir,
                args=args,
            ): index
            for index, configs in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            timing = "" if result.median_ms is None else f" {result.median_ms:.6f} ms"
            ratio = (
                ""
                if result.flex_to_compiled_eager is None
                else f" ratio={result.flex_to_compiled_eager:.6f}"
            )
            print(
                f"pass={pass_index} flex={flex_index} candidate={result.index} "
                f"gpu={result.device} {result.status}{timing}{ratio}",
                flush=True,
            )

    results.sort(key=lambda result: result.index)
    _write_json(
        output_dir / "summary.json",
        [dataclasses.asdict(result) for result in results],
    )
    passed = [result for result in results if result.flex_to_compiled_eager is not None]
    if not passed:
        raise RuntimeError(
            f"no valid configuration for {args.case} FlexGEMM {flex_index}; "
            f"see {output_dir}"
        )
    verification_inputs = []
    for device in args.devices:
        device_results = sorted(
            (result for result in passed if result.device == device),
            key=lambda result: result.flex_to_compiled_eager,
        )
        verification_inputs.extend(device_results[: args.verify_per_device])
    verification_configs = _deduplicate_config_sets(
        tuple(result.configs for result in verification_inputs)
    )
    if not verification_configs:
        raise RuntimeError("no candidates selected for reference-GPU verification")

    verification_dir = output_dir / "reference_verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    reference_device_queue: queue.Queue[int] = queue.Queue()
    reference_device_queue.put(args.devices[0])
    verified = []
    for index, configs in enumerate(verification_configs):
        attempts = []
        for attempt in range(args.verify_retries):
            result = _run_candidate(
                index=index + attempt * len(verification_configs),
                configs=configs,
                device_queue=reference_device_queue,
                output_dir=verification_dir,
                args=args,
            )
            attempts.append(result)
            if result.median_ms is not None:
                break
        result = attempts[-1]
        verified.append(result)
        timing = "" if result.median_ms is None else f" {result.median_ms:.6f} ms"
        ratio = (
            ""
            if result.flex_to_compiled_eager is None
            else f" ratio={result.flex_to_compiled_eager:.6f}"
        )
        print(
            f"verify pass={pass_index} flex={flex_index} candidate={index} "
            f"gpu={result.device} {result.status}{timing}{ratio}",
            flush=True,
        )
    _write_json(
        verification_dir / "summary.json",
        [dataclasses.asdict(result) for result in verified],
    )
    verified_passed = [
        result for result in verified if result.flex_to_compiled_eager is not None
    ]
    if not verified_passed:
        raise RuntimeError(
            f"all reference-GPU verification runs failed for {args.case} "
            f"FlexGEMM {flex_index}; see {verification_dir}"
        )
    winner = min(
        verified_passed,
        key=lambda result: result.median_ms,
    )
    assert winner.median_ms is not None
    assert winner.flex_to_compiled_eager is not None
    print(
        f"winner pass={pass_index} flex={flex_index}: "
        f"{winner.median_ms:.6f} ms ratio={winner.flex_to_compiled_eager:.6f} "
        f"{winner.configs[flex_index]}",
        flush=True,
    )
    return winner.configs


def _run_final(
    configs: tuple[Config, ...],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[CandidateResult, ...]:
    final_args = argparse.Namespace(**vars(args))
    final_args.warmup = args.final_warmup
    final_args.rounds = args.final_rounds
    final_args.iterations = args.final_iterations
    final_args.timeout = args.final_timeout
    final_args.resume = False
    device_queue: queue.Queue[int] = queue.Queue()
    device_queue.put(args.devices[0])
    attempts = []
    for index in range(args.final_retries):
        result = _run_candidate(
            index=index,
            configs=configs,
            device_queue=device_queue,
            output_dir=output_dir / "final",
            args=final_args,
        )
        attempts.append(result)
        if result.median_ms is not None:
            break
        print(
            f"final attempt {index + 1}/{args.final_retries} failed: "
            f"{result.error}",
            flush=True,
        )
    return tuple(attempts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="671b")
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--devices",
        default="0,1,2,3",
        help="comma-separated physical CUDA device indices",
    )
    parser.add_argument("--search", choices=("priority", "full"), default="priority")
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="limit the selected search space for debugging",
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--flex-index",
        type=int,
        action="append",
        help="tune only this FlexGEMM index; repeat to select multiple indices",
    )
    parser.add_argument(
        "--base-config",
        type=_parse_config,
        action="append",
        default=[],
        help="starting JSON config; repeat once per FlexGEMM",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--verify-per-device",
        type=int,
        default=3,
        help="remeasure this many candidates per search GPU on the first device",
    )
    parser.add_argument("--verify-retries", type=int, default=3)
    parser.add_argument("--final-warmup", type=int, default=25)
    parser.add_argument("--final-rounds", type=int, default=10)
    parser.add_argument("--final-iterations", type=int, default=200)
    parser.add_argument("--final-timeout", type=int, default=900)
    parser.add_argument("--final-retries", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/coda_fusion_microbench/autotune"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    if args.case not in SUITES[args.suite]:
        parser.error(
            f"--case must be one of {', '.join(SUITES[args.suite])} "
            f"for --suite {args.suite}"
        )
    try:
        args.devices = tuple(int(device) for device in args.devices.split(","))
    except ValueError as error:
        parser.error(f"--devices must contain integers: {error}")
    if not args.devices:
        parser.error("--devices cannot be empty")
    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    if args.verify_per_device <= 0:
        parser.error("--verify-per-device must be positive")
    if args.verify_retries <= 0:
        parser.error("--verify-retries must be positive")
    if args.final_retries <= 0:
        parser.error("--final-retries must be positive")
    for name in ("warmup", "rounds", "iterations", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = _parse_args()
    case = SUITES[args.suite][args.case]
    current_configs = _initial_configs(args)
    candidates = _search_configs(args.search, args.max_candidates)
    flex_indices = (
        tuple(args.flex_index)
        if args.flex_index is not None
        else tuple(range(case.num_flex_gemms))
    )
    if any(index < 0 or index >= case.num_flex_gemms for index in flex_indices):
        raise ValueError(
            f"FlexGEMM indices for {args.case} must be in "
            f"[0, {case.num_flex_gemms})"
        )

    output_dir = args.output_dir / args.case
    if args.suite != "671b":
        output_dir = args.output_dir / args.suite / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    device_queue: queue.Queue[int] = queue.Queue()
    for device in args.devices:
        device_queue.put(device)

    print(
        f"tuning {args.suite}/{args.case}: {len(candidates)} candidates, "
        f"FlexGEMMs={flex_indices}, GPUs={args.devices}",
        flush=True,
    )
    for pass_index in range(args.passes):
        for flex_index in flex_indices:
            current_configs = _tune_flex_gemm(
                flex_index=flex_index,
                pass_index=pass_index,
                current_configs=current_configs,
                candidates=candidates,
                device_queue=device_queue,
                root_output_dir=output_dir,
                args=args,
            )

    _write_json(output_dir / "best_configs.json", current_configs)
    if args.skip_final:
        return
    (output_dir / "final").mkdir(parents=True, exist_ok=True)
    final_attempts = _run_final(current_configs, output_dir, args)
    _write_json(
        output_dir / "final_summary.json",
        [dataclasses.asdict(result) for result in final_attempts],
    )
    final = final_attempts[-1]
    if final.median_ms is None:
        raise RuntimeError(
            f"final benchmark failed: {final.error}; see {final.log_path}"
        )
    print(f"final: {final.median_ms:.6f} ms; result={final.result_path}")


if __name__ == "__main__":
    main()
