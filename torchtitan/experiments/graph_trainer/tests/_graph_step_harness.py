# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A test-owned copy of ``GraphTrainer``'s traced fwd+bwd step.

This mirrors ``GraphTrainer._make_fx_forward_backward_step``: trace the joint
fwd+loss+bwd graph, run the **full default pass pipeline** over it, then step it
and accumulate gradients. The pass pipeline is the part that must not be
shortcut -- the memory policy reads the graph that dead-code elimination,
canonicalization, FSDP unshard-chain dedup and remat leave behind, and solving a
raw trace instead reports budgets as infeasible that production satisfies.

Two things are deliberately different from the trainer:

* The step loop drops its references to the graph outputs (``del outputs,
  grads``). The trainer leaves them reachable past the return, which keeps the
  previous step's gradients -- one full copy of the model in fp32 -- resident
  into the next step and inflates the measured peak by an amount that has
  nothing to do with the plan. A subclass cannot reach another frame's locals,
  so measuring the plan requires owning the loop.
* Precompile artifacts, in-graph gradient accumulation and pipeline parallelism
  are dropped. None of them affect what is being measured, and each would pull
  in trainer state the harness would then have to fake.

Everything else -- the tracer, the pass list, ``GraphRunner``, the gradient
accumulation call -- is the production code path, imported rather than copied,
so this cannot silently drift from what it is meant to represent.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    annotate_module_fqns,
    apply_simple_fsdp,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    AutoSacConfig,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as llama3_registry,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
)
from torchtitan.experiments.graph_trainer.runner import GraphRunner
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step


logger = logging.getLogger(__name__)

GIB = 1 << 30


def build_compile_config(
    *,
    memory_policy: str,
    memory_budget_gb: float = -1.0,
    solver_type: str = "greedy",
    cpu_offload_bw: int = 20,
) -> GraphTrainerCompileConfig:
    """The real config object, so the pass pipeline sees production defaults.

    cudagraph is disabled: it is orthogonal to the memory policy and its
    replay path holds extra buffers that would muddy a peak measurement.
    """
    return GraphTrainerCompileConfig(
        enable=True,
        mode="aot_fx_trace",
        memory_policy=memory_policy,
        disable_passes=["cudagraph_pass"],
        auto_sac=AutoSacConfig(
            memory_budget_gb=memory_budget_gb,
            solver_type=solver_type,
            cpu_offload_bw=cpu_offload_bw,
            debug_solver=True,
        ),
    )


class GraphStepHarness:
    """Trace, apply every graph pass, and step -- owning the references."""

    def __init__(
        self,
        *,
        model_flavor: str,
        parallel_dims: ParallelDims,
        loss_fn,
        attn_backend: str = "flex",
    ) -> None:
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.model_spec = llama3_registry(model_flavor, attn_backend=attn_backend)

        with torch.device("meta"):
            model = self.model_spec.model.build()
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.init_states(buffer_device=None)
        model.train()
        annotate_module_fqns(model)
        self.model = apply_simple_fsdp(
            model, parallel_dims=parallel_dims, training=TrainingConfig()
        )
        self.params = tuple(p for p in self.model.parameters() if p.requires_grad)

    # -- inputs -----------------------------------------------------------

    def make_inputs(self, num_tokens: int, seq_len: int, vocab: int):
        """Build call args through the model's own ``preprocess_inputs``.

        That is what constructs the FlexInnerAttention BlockMask; feeding the
        model raw tensors fails with "attention_masks must be BlockMask".
        """
        tokens = torch.randint(0, vocab, (num_tokens,), device="cuda")
        label_ids = torch.randint(0, vocab, (num_tokens,), device="cuda")
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32) % seq_len
        inputs, labels, extra_kwargs = self.model.preprocess_inputs(
            {"input": tokens, "positions": positions, "labels": label_ids},
            parallel_dims=self.parallel_dims,
            parallelism=SimpleNamespace(
                pipeline_parallel_degree=1,
                fsdp_reshard_after_forward="default",
            ),
        )
        global_valid_tokens = torch.tensor(
            float(label_ids.numel()), dtype=torch.float, device="cuda"
        )
        return (inputs, labels, global_valid_tokens, extra_kwargs)

    # -- trace + passes ---------------------------------------------------

    def trace_and_compile(self, call_args, compile_config):
        """Trace the joint step and run the full default pass pipeline."""
        maybe_register_blockmask_pytree_node()
        fwd_bwd_fn = make_fwd_bwd_step(self.model, self.loss_fn)
        trace_context = dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims, spmd_typechecking=False
        )
        with trace_context():
            traced = minimal_fx_tracer(fwd_bwd_fn, module=self.model)(*call_args)

        config = SimpleNamespace(
            compile=compile_config,
            parallelism=SimpleNamespace(
                fsdp_reshard_after_forward="default",
                pipeline_parallel_degree=1,
            ),
            optimizer=None,
            activation_checkpoint=None,
            training=TrainingConfig(),
            dataloader=SimpleNamespace(max_num_documents=None),
            model_spec=SimpleNamespace(model=self.model_spec.model),
        )
        passes = construct_default_graph_passes(
            traced,
            config,
            parallel_dims=self.parallel_dims,
            model_parts=[self.model],
        )
        traced.gm = apply_graph_passes(
            traced.gm,
            traced.example_inputs,
            passes,
            compile_config=compile_config,
            respect_disable_passes=True,
        )
        return traced

    # -- run --------------------------------------------------------------

    def run_and_measure(
        self, traced, call_args, *, warmup_steps: int = 2, measured_steps: int = 3
    ) -> float:
        """Step the graph; return peak allocated GiB, max across ranks.

        ``del outputs, grads`` is the point of owning this loop: without it the
        previous step's gradients stay resident through the next step.
        """
        runner = GraphRunner(traced, module=self.model)

        for step in range(warmup_steps + measured_steps):
            for parameter in self.params:
                parameter.grad = None
            if step == warmup_steps:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            outputs = runner(*call_args)
            loss = outputs[0]
            grads = outputs[1:]
            accumulate_param_grads_(self.params, grads)
            del outputs, grads, loss

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del runner

        # Per-rank budget: a peak any rank exceeds is a peak that was not met,
        # so take the max rather than an average.
        buf = torch.tensor([float(peak)], device="cuda", dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(buf, op=dist.ReduceOp.MAX)
        return buf[0].item() / GIB
