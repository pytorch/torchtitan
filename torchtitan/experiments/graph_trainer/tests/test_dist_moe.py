# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer-specific Dist-MoE adapter tests."""

import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch

from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine


def test_graph_trainer_imports_do_not_require_dist_moe() -> None:
    """GraphTrainer and its recipes retain the optional package boundary."""
    script = r"""
import importlib.abc
import sys

class BlockDistMoe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "dist_moe" or fullname.startswith("dist_moe."):
            raise ModuleNotFoundError("blocked optional import", name=fullname)
        return None

sys.meta_path.insert(0, BlockDistMoe())
import torchtitan.experiments.graph_trainer.graph_builder
import torchtitan_recipes.graph_trainer.deepseek_v3
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_graph_engine_supplies_dist_moe_graph_pp_registration() -> None:
    """GraphTrainer supplies liveness and registration before graph tracing."""
    runtime = Mock()
    runtime_config = Mock()
    runtime_config.build.return_value = runtime
    graph_runtime = cast(Any, object.__new__(GraphRuntime))
    graph_runtime._liveness_schedule = SimpleNamespace()
    graph_runtime._graph_pp_ready = False
    graph_runtime._dist_moe_forward_context = None

    engine = cast(Any, object.__new__(GraphTrainingEngine))
    engine.config = SimpleNamespace(
        dist_moe=runtime_config,
        sdc_replayer=None,
        training=SimpleNamespace(
            num_tokens_per_train_step=-1,
            num_tokens_per_microbatch_per_dp_rank=8,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
        ),
        parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=False),
        compile=SimpleNamespace(memory_policy="save_all"),
    )
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    engine.pp_schedule = graph_runtime
    engine.device = torch.device("cuda")
    engine._dist_moe_runtime = None
    engine._forward_backward_body = Mock()

    with patch(
        "torchtitan.experiments.graph_trainer.trainer._maybe_apply_numa_binding"
    ):
        engine._initialize_forward_backward()

    assert engine._dist_moe_runtime is runtime
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is (
        graph_runtime.pipeline_liveness_schedule
    )
    assert runtime_config.build.call_args.kwargs["wgrad_dtype"] is torch.bfloat16
    setter = runtime_config.build.call_args.kwargs["set_forward_context"]
    assert setter.__self__ is graph_runtime
    assert setter.__func__ is GraphRuntime.set_dist_moe_forward_context
