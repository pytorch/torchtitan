# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-host pass-rate sampling worker -- stage 1 of "data washing".

Runs the trainee policy over a SHARD of the R2E task pool and records, for every
(task, sample), whether the coding agent solved it. Together with
``aggregate_passrate.py`` (stage 2, pure CPU) this implements a
rejection-sampling recipe: estimate each task's pass-rate by running the SAME
policy + grader the trainer will use, K samples per task, then keep only the
learnability band (e.g. 0.2 < pass_rate < 0.7). The band is what unstarves
binary-reward GRPO on sparse R2E (most raw tasks are 0% or 100% -> zero-variance
groups the soft filter drops).

This is an EMBARRASSINGLY-PARALLEL worker: there is NO controller, trainer, mesh,
or weight sync. Each host runs one in-process vLLM ``AsyncLLMEngine`` serving the
policy, one ``AnthropicAdapter``, and grades its shard of (task, sample) attempts
concurrently. Launch N hosts with ``--shard-id 0..N-1 --num-shards N`` (strided
sharding balances the repo-sorted pool) writing to one shared ``--out-dir``;
results are resumable (an attempt whose result JSON already exists is skipped) so
a preempted host just re-runs its missing attempts.

Output: ``<out-dir>/results/<instance_id>/sample_<k>.json`` =
``{instance_id, sample_idx, solved, reward, applied, status, error, num_turns}``.
Feed ``<out-dir>/results`` to ``aggregate_passrate.py --results-dir``.

The single-attempt body mirrors ``rollouter._run_claude_rollout`` but drops all
training-loop coupling (no turn capture for training, no rubric/advantage): boot
a sandbox, run ``claude -p`` against the adapter, ``git_diff``, then
``evaluate_r2e`` -> ``solved`` (binary, the default training reward).

Example (one host, smoke)::

    SWE_PROMPT_DATA=.../r2e_subset_4p5k.jsonl DAYTONA_API_KEY=... \
    python -m torchtitan.experiments.rl.examples.swe_r2e.curate_passrate \
        --model .../Qwen3-32B --out-dir /mnt/<bucket>/.../curate_out \
        --shard-id 0 --num-shards 90 --k 8 --tensor-parallel 8 --concurrency 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid

# AsyncLLMEngine (vLLM v1) runs its EngineCore in a child process. The parent has
# already initialized CUDA (torch import / renderer), so the child must SPAWN, not
# fork, or it dies with "Cannot re-initialize CUDA in forked subprocess". Set this
# BEFORE importing vllm so the engine's process-launch picks it up.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # noqa: E402

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.examples.swe_r2e.data import SWER2EDataset, SWER2ESample
from torchtitan.experiments.rl.examples.swe_r2e.grading import evaluate_r2e
from torchtitan.experiments.rl.harness import (
    AnthropicAdapter,
    boot_agent_sandbox,
    git_diff,
    run_claude_code,
    run_host_loop,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.types import Completion

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("curate_passrate")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val and val.strip() else default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        default=os.environ.get("SWE_PROMPT_DATA", ""),
        help="R2E task pool JSONL (default $SWE_PROMPT_DATA).",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get(
            "MODEL", "torchtitan/experiments/rl/example_checkpoint/Qwen3-32B"
        ),
        help="HF checkpoint to sample with (MUST match the trainee policy).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Shared output dir; per-attempt results go to <out-dir>/results/.",
    )
    ap.add_argument("--shard-id", type=int, default=_env_int("SWE_SHARD_ID", 0))
    ap.add_argument("--num-shards", type=int, default=_env_int("SWE_NUM_SHARDS", 1))
    ap.add_argument(
        "--k",
        type=int,
        default=_env_int("SWE_CURATE_K", 8),
        help="Samples per task (pass-rate resolution ~ 1/k).",
    )
    ap.add_argument(
        "--tensor-parallel",
        type=int,
        default=_env_int("SWE_TP", 8),
        help="vLLM tensor-parallel size (8 for a 32B host).",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=_env_int("SWE_ROLLOUT_CONCURRENCY", 24),
        help="Max concurrently-active (boot+agent+grade) attempts on this host.",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--renderer",
        default=os.environ.get("SWE_RENDERER", "qwen3"),
        help="Renderer/chat-template name (e.g. qwen3, qwen3.6). Must match model.",
    )
    ap.add_argument(
        "--agent-mode",
        default=os.environ.get("SWE_AGENT_MODE", "in_sandbox_claude"),
        choices=["in_sandbox_claude", "host_loop"],
        help="in_sandbox_claude = Claude Code CLI bridge (large system prompt); "
        "host_loop = lean host-side ReAct agent (small prompt, matches training).",
    )
    ap.add_argument(
        "--vllm-additional-config",
        default=os.environ.get("SWE_VLLM_ADDITIONAL_CONFIG", ""),
        help='JSON forwarded to vLLM additional_config, e.g. {"gdn_prefill_backend": "triton"} for GDN hybrids.',
    )
    ap.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        default=os.environ.get("SWE_DISABLE_CUSTOM_ALL_REDUCE", "") not in ("", "0"),
        help="Fall back to NCCL all-reduce (some archs hang on custom all-reduce at TP>1).",
    )
    ap.add_argument(
        "--max-model-len", type=int, default=_env_int("SWE_MAX_MODEL_LEN", 24576)
    )
    ap.add_argument(
        "--max-context-len", type=int, default=_env_int("SWE_MAX_CONTEXT_LEN", 22528)
    )
    ap.add_argument(
        "--max-gen-len", type=int, default=_env_int("SWE_MAX_GEN_LEN", 4096)
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("SWE_GPU_MEM", "0.85")),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=_env_int("SWE_CURATE_LIMIT", 0),
        help="Cap tasks in this shard (0 = all); for smoke tests.",
    )
    ap.add_argument(
        "--dump-dir",
        default=os.environ.get("SWE_DUMP_DIR", ""),
        help="If set, write a decoded transcript + final diff per attempt here (diagnostics).",
    )
    return ap.parse_args()


def shard_samples(
    samples: list[SWER2ESample], shard_id: int, num_shards: int
) -> list[SWER2ESample]:
    """Strided shard so each host gets a difficulty-balanced slice of the
    repo-sorted pool (contiguous slices would be single-repo)."""
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"need 0 <= shard_id ({shard_id}) < num_shards ({num_shards})")
    return samples[shard_id::num_shards]


def build_engine(args: argparse.Namespace) -> AsyncLLMEngine:
    """One in-process async vLLM engine serving the policy with continuous batching.

    AsyncLLMEngine (the engine behind vLLM's OpenAI server) natively multiplexes
    many concurrent ``generate`` calls, which is what the K-way-per-task fanout
    needs -- unlike the offline ``LLM`` (one batch at a time) the smoke harness uses.
    """
    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,  # robust over decode-throughput; curation is agent-bound
        trust_remote_code=True,
    )
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    if args.vllm_additional_config:
        # e.g. {"gdn_prefill_backend": "triton"} for Qwen3.5/3.6 GDN hybrids.
        kwargs["additional_config"] = json.loads(args.vllm_additional_config)
    engine_args = AsyncEngineArgs(**kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args)


def make_generate_fn(engine: AsyncLLMEngine):
    """A token-in-token-out ``generate_fn`` the AnthropicAdapter calls per turn.

    Logprobs are irrelevant for curation (we only need the agent's actions and the
    final grade), so they are zero-filled -- the adapter/grader never read them.
    """

    async def generate_fn(
        prompt_token_ids, *, request_id, routing_session_id=None, sampling_config=None
    ):
        sc = sampling_config
        sp = SamplingParams(
            temperature=sc.temperature,
            top_p=sc.top_p,
            max_tokens=sc.max_tokens,
            n=1,
            stop_token_ids=sc.stop_token_ids or None,
            logprobs=None,
        )
        # AsyncLLMEngine requires a globally-unique request_id among in-flight
        # requests; the adapter's id can repeat across turns/sessions, so nonce it.
        eng_req_id = f"{request_id}-{uuid.uuid4().hex[:8]}"
        final = None
        async for out in engine.generate(
            {"prompt_token_ids": list(prompt_token_ids)}, sp, eng_req_id
        ):
            final = out
        o = final.outputs[0]
        tok = list(o.token_ids)
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id=request_id,
            token_ids=tok,
            token_logprobs=[0.0] * len(tok),
            finish_reason=o.finish_reason,
        )

    return generate_fn


def _result_path(out_dir: str, instance_id: str, k: int) -> str:
    safe = instance_id.replace("/", "_")
    return os.path.join(out_dir, "results", safe, f"sample_{k}.json")


def _already_done(path: str) -> bool:
    """An attempt is done iff its result JSON exists and parses (atomic-rename
    write below guarantees a present file is complete)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _write_result(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, path)  # atomic; a present file is always complete (resume-safe)


def _dump_trace(
    dump_dir: str,
    sample: SWER2ESample,
    k: int,
    captured,
    diff_text: str,
    tokenizer,
    status: str,
    solved: bool,
    reward: float,
) -> None:
    """Write a human-readable transcript (system+first prompt, each turn's decoded
    completion, and the final git diff) so a low solve rate can be diagnosed."""
    safe = sample.instance_id.replace("/", "_")
    path = os.path.join(dump_dir, f"{safe}__k{k}.txt")
    os.makedirs(dump_dir, exist_ok=True)
    lines = [
        f"instance_id: {sample.instance_id}",
        f"status={status} solved={solved} reward={reward:.2f} turns={len(captured)}",
        "=" * 80,
    ]
    if captured:
        first = tokenizer.decode(captured[0].prompt_token_ids, skip_special_tokens=False)
        lines += ["TURN-0 PROMPT (system + first user):", first, "=" * 80]
    for i, t in enumerate(captured):
        comp = tokenizer.decode(t.completion_token_ids, skip_special_tokens=False)
        lines += [
            f"--- COMPLETION turn={i} (finish={t.finish_reason}, "
            f"{len(t.completion_token_ids)} tok) ---",
            comp,
        ]
    lines += ["=" * 80, "FINAL GIT DIFF:", diff_text or "(empty)"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


async def grade_attempt(
    *,
    adapter: AnthropicAdapter,
    generate_fn,
    sample: SWER2ESample,
    k: int,
    sampling: SamplingConfig,
    sem: asyncio.Semaphore,
    out_dir: str,
    max_context_tokens: int,
    time_budget_sec: int,
    eval_timeout_sec: int,
    guard_sec: int,
    agent_mode: str = "in_sandbox_claude",
    dump_dir: str = "",
    tokenizer=None,
) -> str:
    """Run + grade ONE (task, sample) attempt; write its result JSON; return status.

    Mirrors ``rollouter._run_claude_rollout`` minus training-loop coupling. Always
    writes a result (errors are caught and recorded) so the aggregator sees every
    attempt and a preempted host resumes only the missing ones.
    """
    path = _result_path(out_dir, sample.instance_id, k)
    if _already_done(path):
        return "resumed"

    sid = f"{sample.instance_id}/attempt={k}/{uuid.uuid4().hex[:6]}"
    adapter.open_session(
        sid,
        generate_fn=generate_fn,
        sampling=sampling,
        routing_session_id=sid,
        max_context_tokens=max_context_tokens,
    )

    status = "error"
    reward = 0.0
    solved = False
    applied = False
    error_msg = ""
    num_turns = 0
    diff_text = ""
    await sem.acquire()
    try:
        async with asyncio.timeout(guard_sec):
            async with boot_agent_sandbox(sample.image) as sb:
                if agent_mode == "host_loop":
                    await run_host_loop(
                        sb,
                        workdir=sample.workdir,
                        session_id=sid,
                        adapter_url=adapter.url,
                        time_budget_sec=time_budget_sec,
                        problem_statement=sample.problem_statement,
                        pre_commands=sample.pre_commands,
                    )
                else:
                    await run_claude_code(
                        sb,
                        workdir=sample.workdir,
                        session_id=sid,
                        adapter_url=adapter.url,
                        time_budget_sec=time_budget_sec,
                        problem_statement=sample.problem_statement,
                        pre_commands=sample.pre_commands,
                    )
                diff_text = await git_diff(sb, sample.workdir, tracked_only=True)
            reward, solved, applied = await evaluate_r2e(
                image=sample.image,
                workdir=sample.workdir,
                diff_text=diff_text,
                r2e=sample.r2e,
                pre_commands=sample.pre_commands,
                timeout_sec=eval_timeout_sec,
            )
            status = "completed"
    except (TimeoutError, asyncio.TimeoutError):
        status = "error_timeout"
        error_msg = "wall_clock_timeout"
    except Exception as e:  # noqa: BLE001 -- one bad attempt must not kill the shard
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("[curate] %s sample %d failed", sample.instance_id, k)
    finally:
        sem.release()
        captured = await adapter.finish_session(sid)
        num_turns = len(captured)
        if dump_dir and tokenizer is not None:
            _dump_trace(
                dump_dir, sample, k, captured, diff_text, tokenizer, status, solved, reward
            )

    _write_result(
        path,
        {
            "instance_id": sample.instance_id,
            "sample_idx": k,
            "solved": bool(solved),
            "reward": float(reward),
            "applied": bool(applied),
            "status": status,
            "error": error_msg,
            "num_turns": num_turns,
        },
    )
    logger.info(
        "[curate] %s sample %d: status=%s solved=%s reward=%.2f turns=%d",
        sample.instance_id,
        k,
        status,
        solved,
        reward,
        num_turns,
    )
    return status


async def main() -> None:
    args = parse_args()
    if not args.data:
        raise ValueError("--data (or $SWE_PROMPT_DATA) is required")
    if "DAYTONA_API_KEY" not in os.environ:
        raise ValueError("DAYTONA_API_KEY must be set for the sandbox backend")

    renderer = RendererConfig(name=args.renderer).build(tokenizer_path=args.model)
    stop_ids = renderer.get_stop_token_ids()

    ds = SWER2EDataset.Config(data_path=args.data, seed=42, shuffle=False).build()
    my_samples = shard_samples(ds._samples, args.shard_id, args.num_shards)
    if args.limit > 0:
        my_samples = my_samples[: args.limit]
    logger.info(
        "shard %d/%d: %d of %d tasks, k=%d -> %d attempts (concurrency=%d)",
        args.shard_id,
        args.num_shards,
        len(my_samples),
        len(ds._samples),
        args.k,
        len(my_samples) * args.k,
        args.concurrency,
    )

    engine = build_engine(args)
    generate_fn = make_generate_fn(engine)
    adapter = AnthropicAdapter(
        renderer=renderer,
        host=os.environ.get("SHIM_BIND_HOST", "127.0.0.1"),
        port=_env_int("SHIM_PORT", 18001),
    )
    await adapter.start()

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_gen_len,
        stop_token_ids=stop_ids,
    )
    sem = asyncio.Semaphore(args.concurrency)
    time_budget_sec = _env_int("SWE_TIME_BUDGET_SEC", 1200)
    eval_timeout_sec = _env_int("SWE_EVAL_TIMEOUT_SEC", 400)
    guard_sec = time_budget_sec + eval_timeout_sec + 300

    t0 = time.time()
    results = await asyncio.gather(
        *(
            grade_attempt(
                adapter=adapter,
                generate_fn=generate_fn,
                sample=sample,
                k=k,
                sampling=sampling,
                sem=sem,
                out_dir=args.out_dir,
                max_context_tokens=args.max_context_len,
                time_budget_sec=time_budget_sec,
                eval_timeout_sec=eval_timeout_sec,
                guard_sec=guard_sec,
                agent_mode=args.agent_mode,
                dump_dir=args.dump_dir,
                tokenizer=getattr(renderer, "_tokenizer", None),
            )
            for sample in my_samples
            for k in range(args.k)
        ),
        return_exceptions=True,
    )

    counts: dict[str, int] = {}
    for r in results:
        key = r if isinstance(r, str) else f"exc:{type(r).__name__}"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "tasks": len(my_samples),
        "k": args.k,
        "attempts": len(results),
        "status_counts": counts,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.join(args.out_dir, "shard_summaries"), exist_ok=True)
    with open(
        os.path.join(args.out_dir, "shard_summaries", f"shard_{args.shard_id}.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)
    logger.info("[curate] shard %d DONE: %s", args.shard_id, summary)

    await adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())
