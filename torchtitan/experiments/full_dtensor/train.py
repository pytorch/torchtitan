# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import torch

from torchtitan.train import main, Trainer

from .llama3.parallelize import build_parallelize_inputs_fn, parallelize_buffers


class FullDTensorTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # parallelize_buffers must be called here instead of in parallelize_fn because
        # buffers are re-initialized after parallelize_fn executes. The current buffer
        # initialization creates regular tensors rather than DTensors.
        for m in self.model_parts:
            parallelize_buffers(m, self.parallel_dims)

        self.parallelize_inputs = build_parallelize_inputs_fn(self.parallel_dims)

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        inputs = input_dict["input"]

        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # Arguments like attention_masks must be in a separate dict since extra_inputs
        # are not forwarded to other stages in PP, but extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        # We could let the model perform parallelize_inputs, but calling it here in the
        # trainer preserves the potential of implementing dataloading pipelining,
        # which offloads logic (e.g., CP load balancing) to CPU and overlaps it with
        # the previous forward(). We also need to consider how PP shards inputs along
        # the batch dimension. For now, keep this function callsite in the trainer.
        inputs, labels = self.parallelize_inputs(inputs, labels)

        assert isinstance(inputs, torch.distributed.tensor.DTensor), type(inputs)
        assert isinstance(labels, torch.distributed.tensor.DTensor), type(labels)

        return inputs, labels, extra_inputs, extra_kwargs


if __name__ == "__main__":
    main(FullDTensorTrainer)
