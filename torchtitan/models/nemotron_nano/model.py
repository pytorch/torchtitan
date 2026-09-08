# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Nemotron-3 Nano: Hybrid Mamba-Transformer MoE Model
# Based on NVIDIA's Nemotron-3 architecture with efficient sparse MoE

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

import torchtitan.distributed
import torchtitan.protocols.module
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.attention import AttentionMasksType

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_nparams_and_active_nparams

# --- Mamba-2 Pure PyTorch Chunk Scan Helpers ---

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)

def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum

def mamba2_chunk_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
):
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    # Rearrange into blocks/chunks
    hidden_states, A, B, C = [reshape_into_chunks(tensor, pad_size, chunk_size) for tensor in (hidden_states, A, B, C)]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk
    L = torch.exp(segment_sum(A))
    G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
    M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    import torch.nn.functional as F
    
    # 3. Compute the inter-chunk SSM recurrence
    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states = new_states[:, :-1]

    # 4. Compute output for inter-chunk
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

    # Add output of intra-chunk and inter-chunk
    Y = Y_diag + Y_off

    Y = Y.reshape(batch_size, -1, num_heads, head_dim)

    if pad_size > 0:
        Y = Y[:, :-pad_size, :, :]

    if D is not None:
        Y = Y + hidden_states.reshape(batch_size, -1, num_heads, head_dim)[:, :sequence_length, :, :] * D[None, None, :, None]

    return Y


class NemotronTransformerBlock(TransformerBlock):
    """
    Nemotron-3 Nano TransformerBlock: Hybrid Mamba-Transformer layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        is_mamba_block: bool = False
        mamba_num_heads: int = 128
        mamba_head_dim: int = 64
        mamba_conv_dim: int = 8192
        mamba_input_projection: Linear.Config | None = None
        mamba_output_projection: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.is_mamba_block = config.is_mamba_block
        self.moe_enabled = not self.is_mamba_block and config.moe is not None
        
        self.mamba_num_heads = config.mamba_num_heads
        self.mamba_head_dim = config.mamba_head_dim
        self.mamba_n_groups = 1  # Standard for Nemotron Mamba-2
        self.mamba_state_dim = 128
        self.chunk_size = 256

        if self.is_mamba_block:
            self.norm = config.attention_norm.build()
            assert config.mamba_input_projection is not None
            dim = config.mamba_input_projection.in_features
            d_inner = config.mamba_conv_dim

            # In Mamba-2, in_proj projects x -> (z, x)
            self.in_proj = config.mamba_input_projection.build()
            self.conv1d = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=4,
                groups=d_inner,
                padding=3,
            )
            
            # x_proj outputs dt, B, C for Mamba-2
            x_proj_out = self.mamba_num_heads + 2 * self.mamba_n_groups * self.mamba_state_dim
            self.x_proj = nn.Linear(d_inner, x_proj_out, bias=False)
            self.dt_proj = nn.Linear(self.mamba_num_heads, self.mamba_num_heads, bias=True)
            self.out_proj = config.mamba_output_projection.build()

            # Mamba-2 A and D are scalars per head
            self.A_log = nn.Parameter(torch.zeros(self.mamba_num_heads, dtype=torch.float32))
            self.D = nn.Parameter(torch.ones(self.mamba_num_heads, dtype=torch.float32))

            nn.init.normal_(self.A_log, mean=0.0, std=0.02)
            nn.init.normal_(self.D, mean=1.0, std=0.02)
        else:
            assert config.attention is not None
            self.attention = config.attention.build()
            self.attention_norm = config.attention_norm.build()

            if config.feed_forward is not None:
                assert config.ffn_norm is not None
                self.ffn_norm = config.ffn_norm.build()
                self.feed_forward = config.feed_forward.build()
            elif config.moe is not None:
                assert config.ffn_norm is not None
                self.ffn_norm = config.ffn_norm.build()
                self.moe = config.moe.build()
            else:
                raise ValueError("Either feed_forward or moe must be provided for Transformer blocks")

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        if self.is_mamba_block:
            x_norm = self.norm(x)

            is_flattened = x_norm.dim() == 2
            if is_flattened:
                x_norm = x_norm.unsqueeze(0)

            B, L, D = x_norm.shape

            # 1. in_proj
            xz = self.in_proj(x_norm)
            x_mamba, z = xz.chunk(2, dim=-1)

            # 2. conv1d
            x_mamba = x_mamba.transpose(1, 2)
            x_mamba = self.conv1d(x_mamba)[:, :, :L]
            x_mamba = x_mamba.transpose(1, 2)
            x_mamba = F.silu(x_mamba)

            # 3. x_proj to dt, B, C
            dt_b_c = self.x_proj(x_mamba)
            dt, B_param, C_param = torch.split(
                dt_b_c,
                [
                    self.mamba_num_heads,
                    self.mamba_n_groups * self.mamba_state_dim,
                    self.mamba_n_groups * self.mamba_state_dim
                ],
                dim=-1,
            )
            
            # Reshape B and C for chunk scan
            B_param = B_param.reshape(B, L, self.mamba_n_groups, self.mamba_state_dim)
            C_param = C_param.reshape(B, L, self.mamba_n_groups, self.mamba_state_dim)

            # 4. dt_proj
            dt = F.softplus(self.dt_proj(dt))

            # 5. Continuous A
            A = -torch.exp(self.A_log.float())

            # Reshape x_mamba for Mamba-2
            x_mamba_reshaped = x_mamba.reshape(B, L, self.mamba_num_heads, self.mamba_head_dim)

            # 6. Mamba-2 Chunk Scan
            y = mamba2_chunk_scan(
                x_mamba_reshaped,
                dt,
                A,
                B_param,
                C_param,
                self.chunk_size,
                D=self.D,
            )

            y = y.reshape(B, L, -1)
            y = y.to(x_mamba.dtype)
            y = y * F.silu(z)

            # 8. out_proj
            out = self.out_proj(y)
            if is_flattened:
                out = out.squeeze(0)
            return x + out
        else:
            h = x + self.attention(self.attention_norm(x), attention_masks, positions)

            if hasattr(self, "feed_forward"):
                out = h + self.feed_forward(self.ffn_norm(h))
            else:
                out = h + self.moe(self.ffn_norm(h))

            return out

    def reset_parameters(self) -> None:
        pass

    def parallelize(self, parallel_dims: "torchtitan.distributed.ParallelDims") -> None:
        super().parallelize(parallel_dims)

        if self.is_mamba_block and parallel_dims.spmd_backend == "spmd_types":
            import spmd_types as spmd

            from torchtitan.distributed.spmd_types import (
                set_current_spmd_mesh,
                spmd_distribute_tensor,
            )

            # Mamba blocks instantiate raw PyTorch parameters/modules that torchtitan's
            # Module.parallelize skips. FSDP2 requires *all* parameters to be DTensors
            # (or annotated plain tensors in spmd_types). We Replicate them across the mesh.
            mesh = parallel_dims.spmd_dense_mesh()
            from torchtitan.models.common.decoder_sharding import dense_param_placement

            layout = dense_param_placement(tp=spmd.R)

            def _distribute_and_annotate(module):
                if isinstance(module, torchtitan.protocols.module.Module) and module is not self:
                    return
                for name, param in module.named_parameters(recurse=False):
                    requires_grad = param.requires_grad
                    dtensor = spmd_distribute_tensor(param, mesh, layout)
                    module.register_parameter(
                        name, nn.Parameter(dtensor, requires_grad=requires_grad)
                    )
                    registered = module._parameters[name]
                    with set_current_spmd_mesh(mesh):
                        spmd.assert_type(registered, layout)
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer is not None:
                        dtensor = spmd_distribute_tensor(buffer, mesh, layout)
                        module.register_buffer(name, dtensor)
                        registered = module._buffers[name]
                        with set_current_spmd_mesh(mesh):
                            spmd.assert_type(registered, layout)
                for child in module.children():
                    _distribute_and_annotate(child)

            # Distribute direct parameters (A_log, D) and raw PyTorch layers
            _distribute_and_annotate(self)

class Nemotron3NanoModel(Decoder):
    """
    Nemotron-3 Nano: Hybrid Mamba-Transformer Mixture-of-Experts model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 262144
        num_experts: int = 128
        top_k_experts: int = 6
        max_context_length: int = 1000000
        mamba_num_heads: int = 128
        mamba_head_dim: int = 64
        mamba_conv_dim: int = 8192

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from torchtitan.models.nemotron_nano.sharding import (
                set_nemotron_sharding_config,
            )

            set_nemotron_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            nparams, _ = get_nparams_and_active_nparams(model)
            active_params = nparams * (self.top_k_experts / self.num_experts)
            return nparams, int(6 * active_params)

    def verify_module_protocol(self) -> None:
        pass
