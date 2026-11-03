# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP training and HF conversion when only one rank has an image.

T: text tokens, P: image patches, F: flattened patch features.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.deepseek_v4_1 import build_model_config
from torchtitan.models.deepseek_v4_1.moe import ModalityMoE
from torchtitan.models.deepseek_v4_1.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
)

pytestmark = pytest.mark.multi_gpu


class TestDeepSeekV41MultimodalFSDP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_mixed_rank_images_train_encoder_aligner_and_markers(self):
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        try:
            self._train_mixed_rank_images()
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

    def _train_mixed_rank_images(self):
        parallelism = ParallelismConfig(data_parallel_shard_degree=2)
        context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=True,
        )
        config = build_model_config(
            "debugmodel",
            parallelism=parallelism,
            seq_len=32,
            vision_enabled=True,
            indexer_loss_coeff=0.0,
        )
        with torch.device("meta"):
            model = config.build()
        config.local_compile.regions = []
        model.parallelize(
            parallelism_context=context,
            training=TrainingConfig(
                max_context_length=32, disable_accelerator_graphs=True
            ),
            parallelism=parallelism,
            local_compile=config.local_compile,
            activation_remat=DefaultActivationRematConfig(),
            dump_folder=str(Path(self.file_name).parent),
        )
        with context.activate_spmd():
            model.to_empty(device=self.device_type)
            model.init_states(buffer_device=torch.device(self.device_type))
        model.train()
        tokens_T = torch.arange(18, device=self.device_type)
        types_T = torch.tensor(
            [-1, -1, 0, 1, 1, 2, 1, 1, 2, 3] + [-1] * 8,
            device=self.device_type,
        )
        pixels_PF = torch.randn(36, 12, device=self.device_type, dtype=torch.bfloat16)
        optimizers = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=0.002)]
        ).build(model_parts=[model])
        type(model)._register_optimizer_hooks(optimizers, [model], context)

        def forward():
            return (
                model(
                    tokens_T,
                    token_types=types_T,
                    pixel_values=pixels_PF,
                    grid_hw=((6, 6),),
                )
                if self.rank == 1
                else model(tokens_T)
            )

        losses = []
        for _ in range(10):
            optimizers.zero_grad()
            with context.activate_spmd():
                logits_TV = forward()
                loss = F.cross_entropy(logits_TV.float(), tokens_T + 1)
                loss.backward()
                for parameter in (
                    model.vision.patch_embed.weight,
                    model.aligner.w1.weight,
                    model.image_embeddings.weight,
                ):
                    assert parameter.grad is not None
                    gradient = parameter.grad
                    if isinstance(gradient, DTensor):
                        gradient = gradient.to_local()
                    assert torch.isfinite(gradient).all()
                    assert gradient.float().norm() > 0
                optimizers.step()
                for moe in model.modules():
                    if isinstance(moe, ModalityMoE):
                        assert torch.count_nonzero(moe.router.tokens_per_expert_E) == 0
                        assert (
                            torch.count_nonzero(moe.router.image_tokens_per_expert_E)
                            == 0
                        )
                        biases = [
                            torch.empty_like(moe.expert_bias_vl_E) for _ in range(2)
                        ]
                        torch.distributed.all_gather(biases, moe.expert_bias_vl_E)
                        torch.testing.assert_close(biases[0], biases[1], rtol=0, atol=0)
            losses.append(loss.item())
        assert losses[-1] < losses[0]
        model.eval()
        with context.activate_spmd(), torch.no_grad():
            native = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            adapter = DeepSeekV41StateDictAdapter(config)
            restored = adapter.from_hf(adapter.to_hf(native))
            assert restored.keys() == native.keys()
            for name, original in native.items():
                actual = restored[name]
                torch.testing.assert_close(
                    actual.full_tensor() if isinstance(actual, DTensor) else actual,
                    original.full_tensor()
                    if isinstance(original, DTensor)
                    else original,
                    rtol=0,
                    atol=0,
                )
            expected_TV = forward()
            model.lm_head.weight.zero_()
            set_model_state_dict(model, restored)
            torch.testing.assert_close(forward(), expected_TV, rtol=0, atol=0)
