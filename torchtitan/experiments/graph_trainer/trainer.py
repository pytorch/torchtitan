# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_teardown
from torchtitan.trainer import Trainer


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def train_step(self, *args, **kwargs):
        """Override to handle SymFloat grad_norm from fx.Interpreter execution.

        When graph PP executes subgraphs via fx.Interpreter, gradient
        operations may return SymFloat instead of regular Tensor. The
        base trainer's clip_grad_norm_ can return SymFloat which breaks
        metrics formatting. This override calls the parent and lets it
        handle the error, then the metrics log converts appropriately.
        """
        # Temporarily patch metrics_processor.log to convert grad_norm
        original_log = self.metrics_processor.log

        def patched_log(step, avg_loss, max_loss, grad_norm, **kw):
            # Handle SymFloat/SymInt from fx.Interpreter backward execution
            import torch

            if isinstance(grad_norm, (torch.SymFloat, torch.SymInt)):
                grad_norm = torch.sym_float(grad_norm)
                # Try to get the concrete hint value
                if hasattr(grad_norm, "node") and hasattr(grad_norm.node, "hint"):
                    hint = grad_norm.node.hint
                    grad_norm = hint if hint is not None else float("inf")
                else:
                    grad_norm = float("inf")
            return original_log(step, avg_loss, max_loss, grad_norm, **kw)

        self.metrics_processor.log = patched_log
        try:
            super().train_step(*args, **kwargs)
        finally:
            self.metrics_processor.log = original_log

    def close(self) -> None:
        super().close()

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
