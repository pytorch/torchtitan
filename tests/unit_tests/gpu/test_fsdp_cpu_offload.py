# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
)


pytestmark = pytest.mark.multi_gpu


class _VisionEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class _TinyVLM(nn.Module):
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


def _offload_policy(module: nn.Module) -> object | None:
    """The offload policy of the FSDP group that owns ``module``'s parameters.

    The root group is skipped: every parameter belongs to a nested group here, so
    the root carries no param group of its own.
    """
    state = _get_module_fsdp_state(module)
    assert state is not None, f"{type(module).__name__} was not sharded"
    group = state._fsdp_param_group
    assert group is not None, f"{type(module).__name__} owns no parameters"
    return group.offload_policy


class TestVisionEncoderCPUOffload(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _build(self, cpu_offload: bool) -> _TinyVLM:
        mesh = self.build_device_mesh()
        with torch.device("meta"):
            model = _TinyVLM()

        apply_fsdp_to_vision_encoder(
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
        return model

    @with_comms
    def test_vision_encoder_gets_the_same_offload_policy_as_the_decoder(self) -> None:
        model = self._build(cpu_offload=True)
        assert isinstance(_offload_policy(model.vision_encoder), CPUOffloadPolicy)
        # ... the same policy the decoder's own units get.
        assert isinstance(_offload_policy(model.tok_embeddings), CPUOffloadPolicy)

    @with_comms
    def test_backward_runs_with_cpu_offload_on_a_model_with_a_vision_encoder(
        self,
    ) -> None:
        # Without the encoder's offload policy its sharded parameters stay on
        # CPU while FSDP produces CUDA gradients for them, and backward raises
        # "attempting to assign a gradient with device type 'cuda' to a tensor
        # with device type 'cpu'".
        model = self._build(cpu_offload=True)
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
        model = self._build(cpu_offload=False)
        assert not isinstance(_offload_policy(model.vision_encoder), CPUOffloadPolicy)
        assert not isinstance(_offload_policy(model.tok_embeddings), CPUOffloadPolicy)
