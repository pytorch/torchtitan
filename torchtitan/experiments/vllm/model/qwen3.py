# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

logger = init_logger(__name__)


class TorchTitanQwen3ForCausalLM(Qwen3ForCausalLM):
    """
    TorchTitan-trained Qwen3 dense model adapter for vLLM.

    This class extends the standard Qwen3ForCausalLM to support loading
    weights from TorchTitan checkpoints with different naming conventions.
    The architecture is identical to standard Qwen3 - only weight names differ.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        pass

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        pass
