# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Isolated harness smoke: real Claude Code in a Daytona sandbox <-> the Anthropic
adapter backed by a plain vLLM engine (no Monarch trainer, no weight sync, no
runaway collection loop). Boots exactly ONE sandbox for ONE sample and prints the
captured turns + grade, so the sandbox -> Claude Code -> adapter -> grading path
can be debugged independently of the RL training loop.

    PROMPT_DATA=.../easy_one.jsonl DAYTONA_API_KEY=... \\
        python -m torchtitan.experiments.rl.examples.swe_r2e.local_smoke_harness
"""

from __future__ import annotations

import asyncio
import logging
import os

from vllm import LLM, SamplingParams

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.examples.swe_r2e.data import SWER2EDataset
from torchtitan.experiments.rl.examples.swe_r2e.grading import evaluate_r2e
from torchtitan.experiments.rl.harness import (
    AnthropicAdapter,
    boot_agent_sandbox,
    git_diff,
    run_claude_code,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.types import Completion

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("local_smoke_harness")

MODEL = os.environ.get(
    "MODEL", "torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B"
)
MAX_MODEL_LEN = int(os.environ.get("SWE_MAX_MODEL_LEN", "24576"))
MAX_CONTEXT = int(os.environ.get("SWE_MAX_CONTEXT_LEN", "20480"))
MAX_GEN = int(os.environ.get("MAX_GEN_LEN", "2048"))


async def main() -> None:
    renderer = RendererConfig(name="qwen3").build(tokenizer_path=MODEL)
    stop_ids = renderer.get_stop_token_ids()
    logger.info("renderer=%s stop_ids=%s", type(renderer).__name__, stop_ids)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )

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
            logprobs=0,
        )

        def _run():
            outs = llm.generate(
                {"prompt_token_ids": list(prompt_token_ids)}, sp, use_tqdm=False
            )
            return outs[0].outputs[0]

        o = await asyncio.to_thread(_run)
        tok = list(o.token_ids)
        if o.logprobs:
            lps = [next(iter(d.values())).logprob for d in o.logprobs]
        else:
            lps = [0.0] * len(tok)
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id=request_id,
            token_ids=tok,
            token_logprobs=lps,
            finish_reason=o.finish_reason,
        )

    adapter = AnthropicAdapter(
        renderer=renderer,
        host=os.environ.get("SHIM_BIND_HOST", "127.0.0.1"),
        port=int(os.environ.get("SHIM_PORT", "18031")),
    )
    await adapter.start()

    ds = SWER2EDataset.Config(
        data_path=os.environ["PROMPT_DATA"], seed=42, shuffle=False
    ).build()
    sample = next(iter(ds))
    logger.info(
        "sample=%s image=%s workdir=%s",
        sample.instance_id,
        sample.image,
        sample.workdir,
    )

    sid = "smoke/sample=0"
    sampling = SamplingConfig(
        temperature=1.0, top_p=1.0, max_tokens=MAX_GEN, stop_token_ids=stop_ids
    )
    adapter.open_session(
        sid,
        generate_fn=generate_fn,
        sampling=sampling,
        routing_session_id=sid,
        max_context_tokens=MAX_CONTEXT,
    )

    diff = ""
    try:
        async with boot_agent_sandbox(sample.image) as sb:
            await run_claude_code(
                sb,
                workdir=sample.workdir,
                session_id=sid,
                adapter_url=adapter.url,
                time_budget_sec=int(os.environ.get("SWE_TIME_BUDGET_SEC", "600")),
                problem_statement=sample.problem_statement,
                pre_commands=sample.pre_commands,
            )
            diff = await git_diff(sb, sample.workdir, tracked_only=True)
    finally:
        turns = await adapter.finish_session(sid)

    print("\n===== HARNESS SMOKE RESULT =====")
    print(f"captured turns: {len(turns)}")
    for i, t in enumerate(turns[:12]):
        print(
            f"  turn {i}: prompt={len(t.prompt_token_ids)} "
            f"completion={len(t.completion_token_ids)} "
            f"finish={t.finish_reason} extends={t.extends_previous}"
        )
    print(f"diff length: {len(diff)} chars")
    if diff:
        print("diff head:\n" + diff[:600])

    reward, solved, applied = await evaluate_r2e(
        image=sample.image,
        workdir=sample.workdir,
        diff_text=diff,
        r2e=sample.r2e,
        pre_commands=sample.pre_commands,
        timeout_sec=int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "400")),
    )
    print(f"REWARD={reward} solved={solved} applied={applied}")
    await adapter.stop()


if __name__ == "__main__":
    asyncio.run(main())
