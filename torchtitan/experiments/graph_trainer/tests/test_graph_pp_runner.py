# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest
from typing import Any
from unittest import mock

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    BACKWARD_INPUT,
    FORWARD,
    FULL_BACKWARD,
    OVERLAP_F_B,
)

from torchtitan.config import ParallelismConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    compute_annotated_loss,
    ensure_boxed_graph_module,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp import multiplex_fw_bw_graph
from graph_trainer.graph_pp.graph_builder import (
    _build_graph_pp_overlap_graphs,
    _build_stage_graphs,
    _compile_graph_pp_module,
    _execute_graph_module,
    GraphTrainerStageGraphProvider,
)
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    _validate_graph_pp_config,
)

from graph_trainer.graph_pp.runner import (
    _post_fwd_common,
    _prepare_fwd_user_args,
    GraphPipelineRuntime,
)
from graph_trainer.graph_pp.stage import GraphPPStageRuntimeState
from graph_trainer.graph_pp.utils import (
    normalize_graph_pp_microbatch_inputs,
)
from graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)


def _boxed_run(gm: fx.GraphModule, args: list[object]):
    return fx.Interpreter(gm).boxed_run(args)


def _build_test_stage_graphs(
    stage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
    *,
    compile_graphs: bool = True,
) -> None:
    _build_stage_graphs(
        stage,
        args,
        kwargs,
        target,
        loss_kwargs,
        loss_fn=stage.loss_fn,
        compile_config=stage.compile_config,
        model_config=stage.model_config,
        parallelism=stage.parallelism,
        compile_graphs=compile_graphs,
    )


def _make_test_stage(
    submod: nn.Module,
    *,
    is_last: bool,
    loss_fn=None,
    stage_index: int = 0,
    output_grads=None,
    compile_config: GraphTrainerCompileConfig | None = None,
    runtime_validate: bool = False,
):
    stage = types.SimpleNamespace(
        submod=submod,
        is_last=is_last,
        loss_fn=loss_fn,
        stage_index=stage_index,
        compile_config=compile_config or GraphTrainerCompileConfig(enable_passes=False),
        model_config=None,
        parallelism=None,
        _runtime_validate=runtime_validate,
    )
    if not is_last:
        if output_grads is None:
            raise ValueError("non-last test stages must provide output_grads metadata")
        stage._stage_meta = types.SimpleNamespace(
            outputs=pytree.tree_leaves(output_grads),
            output_grads=pytree.tree_leaves(output_grads),
        )
        stage._to_tensor = lambda meta: meta
    return stage


def _split_batch_offset_block_masks() -> list[Any]:
    from torch.distributed.pipelining.microbatch import _split_block_mask
    from torch.nn.attention.flex_attention import create_block_mask

    maybe_register_blockmask_pytree_node()

    def mask_mod(b, h, q_idx, kv_idx):
        return (b == 1) & (q_idx >= kv_idx)

    block_mask = create_block_mask(
        mask_mod,
        B=2,
        H=None,
        Q_LEN=128,
        KV_LEN=128,
        device="cpu",
    )
    return _split_block_mask(block_mask, 2)


def _trace_mask_mod_replay(mask0: Any, mask1: Any) -> tuple[bool, bool]:
    def forward(mask):
        b = torch.tensor(0)
        h = torch.tensor(0)
        q_idx = torch.tensor(1)
        kv_idx = torch.tensor(0)
        return mask.mask_mod(b, h, q_idx, kv_idx)

    traced = minimal_fx_tracer(forward)(mask0)
    replay = run_traced(traced)
    return replay(mask0).item(), replay(mask1).item()


class GraphPipelineRuntimeTraceTest(unittest.TestCase):












    def test_single_stage_schedule_hard_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime PP schedule"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
            )

    def test_runtime_schedule_validation_accepts_interleaved(self) -> None:
        _validate_graph_pp_config(
            compile_config=GraphTrainerCompileConfig(),
            parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        )

    def test_graph_pp_accepts_zero_two_fsdp_reshard_policies(self) -> None:
        for policy in ("default", "never"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B",
                    fsdp_reshard_after_forward=policy,
                ),
            )

        with self.assertRaisesRegex(ValueError, "fsdp_reshard_after_forward"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(
                    pipeline_parallel_schedule="Interleaved1F1B",
                    fsdp_reshard_after_forward="always",
                ),
            )















    def test_last_stage_chunked_loss_preserves_hidden_grad_accumulator(self) -> None:
        class LastStage(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(16, 16),
                    nn.ReLU(),
                    nn.Linear(16, 16),
                )
                self.lm_head = nn.Linear(16, 33, bias=False)
                self._skip_lm_head = True

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                hidden_states = self.block(x)
                if self._skip_lm_head:
                    return hidden_states
                return self.lm_head(hidden_states)

        torch.manual_seed(0)
        model = LastStage()
        loss_fn = ChunkedLossWrapperWithParamGrads(
            ChunkedLossWrapperWithParamGrads.Config(num_chunks=4)
        )
        loss_fn.set_lm_head(model.lm_head)
        stage = _make_test_stage(
            model,
            is_last=True,
            loss_fn=loss_fn,
            stage_index=1,
        )
        x = torch.randn(16, 16, requires_grad=True)
        labels = torch.randint(0, 33, (16,))
        global_valid_tokens = torch.tensor(float(labels.numel()))

        _build_test_stage_graphs(
            stage,
            (x,),
            {},
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )

        state = list(model.parameters())
        loss, saved = stage.graphs.forward(
            (x,),
            {},
            labels,
            {"global_valid_tokens": global_valid_tokens},
            unsharded_param_values=state,
            flat_buffer_values=[],
        )
        input_grads, param_grads = stage.graphs.full_backward(
            (loss,),
            saved,
            (),
        )

        expected_loss, _ = loss_fn(model(x), labels, global_valid_tokens)
        expected_grads = torch.autograd.grad(
            expected_loss,
            [*model.parameters(), x],
        )
        self.assertTrue(torch.equal(loss, expected_loss))
        for actual, expected in zip(
            param_grads + input_grads, expected_grads, strict=True
        ):
            self.assertTrue(torch.equal(actual, expected))
        self.assertGreater(torch.linalg.vector_norm(input_grads[0]).item(), 0.0)



if __name__ == "__main__":
    unittest.main()
