# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch
import torch.nn as nn
import torchtitan.distributed.fsdp as titan_fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_multimodal_encoder,
)
from torchtitan.models.base.module import Module
from torchtitan.models.muse_spark.flavors import muse_spark_navit_encoder_config
from torchtitan.models.muse_spark.vision_encoder import NaViTEncoder


pytestmark = pytest.mark.multi_gpu


class _VisionEncoder(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _Block(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.feed_forward = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


class _TinyVLM(Module):
    """The module shape ``apply_fsdp_to_decoder`` reads, plus a vision encoder.

    Mirrors how the VLMs (qwen3_5, kimi_k2_7, kimi_k3, muse_glimmer) compose the
    two helpers: the encoder is sharded first as one unit, then the decoder walks
    the embedding / blocks / head and finally shards the root.
    """

    def __init__(self, dim: int = 32, vocab: int = 64, n_layers: int = 2):
        super().__init__()
        self.enable_weight_tying = False
        self.tok_embeddings = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleDict(
            {str(i): _Block(dim) for i in range(n_layers)},
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.vision_encoder = _VisionEncoder(dim)

    def forward(self, tokens: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens) + self.vision_encoder(pixels)
        for block in self.layers.values():
            h = block(h)
        return self.lm_head(self.norm(h))

    def init_weights(self) -> None:
        for p in self.parameters():
            nn.init.normal_(p, std=0.02)


def _vision_inputs(
    encoder: NaViTEncoder,
    *,
    has_image: bool,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_size = encoder.downsample_factor
    pixel_values = (
        torch.randn(
            grid_size**2,
            encoder.conv1_linear.in_features,
            device=device_type,
            dtype=torch.bfloat16,
        )
        if has_image
        else torch.zeros(
            grid_size**2,
            encoder.conv1_linear.in_features,
            device=device_type,
        )
    )
    grid_thw = torch.tensor(
        [[1, grid_size, grid_size]],
        device=device_type,
        dtype=torch.int64,
    )
    return pixel_values, grid_thw


class TestVisionEncoderCPUOffload(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _build(self, cpu_offload: bool) -> tuple[_TinyVLM, dict[int, object]]:
        """Build the model and return the offload policy that torchtitan passes
        to ``fully_shard`` for each module, by module id."""
        mesh = self.build_device_mesh()
        with torch.device("meta"):
            model = _TinyVLM()

        policies: dict[int, object] = {}
        fully_shard = titan_fsdp.fully_shard

        def recording_fully_shard(module, *args, **kwargs):
            for m in module if isinstance(module, list) else [module]:
                policies[id(m)] = kwargs.get("offload_policy")
            return fully_shard(module, *args, **kwargs)

        with mock.patch.object(titan_fsdp, "fully_shard", recording_fully_shard):
            apply_fsdp_to_multimodal_encoder(
                model.vision_encoder,
                mesh,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cpu_offload=cpu_offload,
            )
            apply_fsdp_to_decoder(
                model,
                mesh,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                pp_enabled=False,
                cpu_offload=cpu_offload,
            )
        # trainer.py materializes the whole model on CPU under cpu offload.
        model.to_empty(device="cpu" if cpu_offload else self.device_type)
        with torch.no_grad():
            model.init_weights()
        return model, policies

    @with_comms
    def test_vision_encoder_gets_the_same_offload_policy_as_the_decoder(self) -> None:
        model, policies = self._build(cpu_offload=True)
        assert isinstance(policies[id(model.vision_encoder)], CPUOffloadPolicy)
        # ... the same policy the decoder's own units get.
        assert isinstance(policies[id(model.tok_embeddings)], CPUOffloadPolicy)

    @with_comms
    def test_backward_runs_with_cpu_offload_on_a_model_with_a_vision_encoder(
        self,
    ) -> None:
        # Without the encoder's offload policy its sharded parameters stay on
        # CPU while FSDP produces CUDA gradients for them, and backward raises
        # "attempting to assign a gradient with device type 'cuda' to a tensor
        # with device type 'cpu'".
        model, _ = self._build(cpu_offload=True)
        tokens = torch.randint(0, 64, (2, 4), device=self.device_type)
        pixels = torch.randn(2, 4, 32, device=self.device_type, dtype=torch.bfloat16)
        model(tokens, pixels).sum().backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert (
                param.grad.device == param.device
            ), f"{name}: grad on {param.grad.device}, param on {param.device}"

    @with_comms
    def test_no_offload_policy_when_cpu_offload_is_off(self) -> None:
        model, policies = self._build(cpu_offload=False)
        assert not isinstance(policies[id(model.vision_encoder)], CPUOffloadPolicy)
        assert not isinstance(policies[id(model.tok_embeddings)], CPUOffloadPolicy)


class TestConditionalVisionFSDP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _run_mixed_rank_image_presence(self, *, cpu_offload: bool) -> None:
        mesh = self.build_device_mesh()
        with torch.device("meta"):
            config = muse_spark_navit_encoder_config(
                latent_dim=8,
                num_layers=0,
                num_heads=2,
                mlp_hidden_dim=16,
                patch_size=2,
                patch_temporal=1,
                downsample_factor=2,
                sparse_attention_factor=1,
                pos_emb_grid_h=2,
                pos_emb_grid_w=2,
            )
            config.pos_emb_grid_order = "raster"
            encoder = config.build()
        apply_fsdp_to_multimodal_encoder(
            encoder,
            mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            reshard_after_forward_policy="always",
            cpu_offload=cpu_offload,
        )
        encoder.to_empty(device="cpu" if cpu_offload else self.device_type)
        with torch.no_grad():
            for parameter in encoder.parameters():
                nn.init.normal_(parameter, std=0.02)
            encoder.rope_freq.init_states(
                buffer_device=torch.device(self.device_type) if cpu_offload else None
            )

        has_image = self.rank == 0
        pixel_values, grid_thw = _vision_inputs(
            encoder,
            has_image=has_image,
            device_type=self.device_type,
        )

        output_TO = encoder(pixel_values, grid_thw=grid_thw)
        self.assertEqual(output_TO.device.type, self.device_type)
        (output_TO.sum() * int(has_image)).backward()

        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertEqual(parameter.grad.device, parameter.device)

    @with_comms
    def test_mixed_rank_image_presence_completes_backward(self) -> None:
        self._run_mixed_rank_image_presence(cpu_offload=False)

    @with_comms
    def test_mixed_rank_image_presence_completes_backward_with_cpu_offload(
        self,
    ) -> None:
        self._run_mixed_rank_image_presence(cpu_offload=True)


class TestVisionHSDP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_image_then_empty_accumulation(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        with torch.device("meta"):
            config = muse_spark_navit_encoder_config(
                latent_dim=8,
                num_layers=0,
                num_heads=2,
                mlp_hidden_dim=16,
                patch_size=2,
                patch_temporal=1,
                downsample_factor=2,
                sparse_attention_factor=1,
                pos_emb_grid_h=2,
                pos_emb_grid_w=2,
            )
            config.pos_emb_grid_order = "raster"
            encoder = config.build()
        apply_fsdp_to_multimodal_encoder(
            encoder,
            mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            reshard_after_forward_policy="always",
        )
        encoder.to_empty(device=self.device_type)
        with torch.no_grad():
            for parameter in encoder.parameters():
                nn.init.normal_(parameter, std=0.02)
            encoder.rope_freq.init_states()

        for microbatch_index, image_active in enumerate((True, False)):
            encoder.set_requires_all_reduce(microbatch_index == 1)
            has_image = image_active and self.rank == 0
            pixel_values, grid_thw = _vision_inputs(
                encoder,
                has_image=has_image,
                device_type=self.device_type,
            )
            output_TO = encoder(pixel_values, grid_thw=grid_thw)
            (output_TO.float().sum() * int(has_image)).backward()

        self.assertTrue(
            all(parameter.grad is not None for parameter in encoder.parameters())
        )
