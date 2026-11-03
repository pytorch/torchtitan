# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP training and HF conversion when only one rank has an image.

T: text tokens, P: image patches, F: flattened patch features, B: draft blocks,
S: draft positions, D: hidden width, V: vocabulary.
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
            num_draft_layers=3,
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

        def forward(*, return_draft_context=False):
            return (
                model(
                    tokens_T,
                    token_types=types_T,
                    pixel_values=pixels_PF,
                    grid_hw=((6, 6),),
                    return_draft_context=return_draft_context,
                )
                if self.rank == 1
                else model(tokens_T, return_draft_context=return_draft_context)
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
            _, context_TD = forward(return_draft_context=True)
            anchors_B = torch.tensor([5, 10], device=self.device_type)
            draft_ids_BS = torch.full(
                (2, 5), model.drafter.config.noise_token_id, device=self.device_type
            )
            draft_ids_BS[:, 0] = tokens_T[anchors_B]
            embedding_weight_VD = model.tok_embeddings.weight
            output_weight_VD = model.lm_head.weight
            if isinstance(embedding_weight_VD, DTensor):
                embedding_weight_VD = embedding_weight_VD.full_tensor()
            if isinstance(output_weight_VD, DTensor):
                output_weight_VD = output_weight_VD.full_tensor()
            embeddings_TD = F.embedding(
                draft_ids_BS.flatten(), embedding_weight_VD.to(torch.bfloat16)
            )
            # FSDP can release an all-gather view after the next backward.
            output_weight_VD = output_weight_VD.detach().clone().to(torch.bfloat16)
        optimizers.zero_grad()
        model.drafter.train()
        draft_optimizers = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=0.002)]
        ).build(model_parts=[model.drafter])
        type(model.drafter)._register_optimizer_hooks(
            draft_optimizers, [model.drafter], context
        )
        previous_BS = torch.arange(10, device=self.device_type).view(2, 5)
        main_positions_T = torch.arange(18, device=self.device_type)
        draft_losses = []
        for _ in range(10):
            draft_optimizers.zero_grad()
            with context.activate_spmd():
                logits_BSV, confidence_BS = model.forward_draft(
                    context_TD,
                    embeddings_TD,
                    previous_BS,
                    anchors_B,
                    main_positions_T=main_positions_T,
                    output_weight_VD=output_weight_VD,
                )
                draft_loss = F.cross_entropy(
                    logits_BSV.float().flatten(0, 1), (previous_BS + 1).flatten()
                ) + F.binary_cross_entropy_with_logits(
                    confidence_BS, torch.full_like(confidence_BS, 0.5)
                )
                draft_loss.backward()
                for parameter in (
                    model.drafter.layers["0"].main_proj.weight,
                    model.drafter.layers["2"].markov_head.embed.weight,
                    model.drafter.layers["2"].confidence_head.proj.weight,
                ):
                    assert parameter.grad is not None
                    gradient = parameter.grad
                    if isinstance(gradient, DTensor):
                        gradient = gradient.full_tensor()
                    assert torch.isfinite(gradient).all()
                    assert gradient.float().norm() > 0
                draft_optimizers.step()
                for layer in model.drafter.layers.values():
                    moe = layer.feed_forward
                    assert torch.count_nonzero(moe.router.tokens_per_expert_E) == 0
                    biases = [torch.empty_like(moe.expert_bias_E) for _ in range(2)]
                    torch.distributed.all_gather(biases, moe.expert_bias_E)
                    torch.testing.assert_close(biases[0], biases[1], rtol=0, atol=0)
            draft_losses.append(draft_loss.item())
        assert draft_losses[-1] < draft_losses[0]
        with context.activate_spmd(), torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.startswith("drafter."):
                    assert parameter.grad is None, name
                    original = native[name]
                    torch.testing.assert_close(
                        parameter.full_tensor()
                        if isinstance(parameter, DTensor)
                        else parameter,
                        original.full_tensor()
                        if isinstance(original, DTensor)
                        else original,
                        rtol=0,
                        atol=0,
                    )
