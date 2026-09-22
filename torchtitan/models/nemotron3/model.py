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
from torch.distributed.tensor import distribute_tensor, DTensor
from torch import nn

import torchtitan.distributed
import torchtitan.protocols.module
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.attention import AttentionMasksType

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)

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

    # The D skip connection is a true residual on the *undiscretized* input, so it
    # must be captured before x is scaled by dt below (reference:
    # transformers Mamba2Mixer.torch_forward computes D_residual first).
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

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

    # Add the D residual (computed pre-discretization) while still padded, then trim.
    if D_residual is not None:
        Y = Y + D_residual.reshape(batch_size, -1, num_heads, head_dim)

    if pad_size > 0:
        Y = Y[:, :-pad_size, :, :]

    return Y


class NoRoPE(RoPE):
    """Identity position embedding.

    Nemotron-H applies no rotary embedding on its attention layers -- positional
    information is carried by the interleaved Mamba layers instead
    (``NemotronHAttention.forward`` never calls ``apply_rotary_pos_emb``). The
    attention module always invokes ``self.rope``, so this supplies a no-op.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        dim: int = 0
        max_context_length: int = 0

        # No build() override, for the same reason as NemotronMLP: it would
        # bypass Module.Config.build and drop _param_init/_sharding_config.
        # Harmless for a no-op module today, but it is the same latent trap.

    def _precompute_cache(self) -> torch.Tensor:
        return torch.empty(0)

    def forward(self, xq, xk, positions=None):
        return xq, xk


class MambaRMSNormGated(nn.Module):
    """Group-wise gated RMSNorm applied to the SSM output before ``out_proj``.

    Mirrors transformers' ``Zamba2RMSNormGated`` as used by Nemotron-H: the gate
    is applied first, then RMS normalization is computed per group of channels.
    """

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_size = group_size
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))

        *prefix, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(*prefix, channels // self.group_size, self.group_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.reshape(*prefix, channels)

        return self.weight * hidden_states.to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class NemotronMLP(torchtitan.protocols.module.Module):
    """Nemotron-H feed-forward: ungated ``down_proj(relu2(up_proj(x)))``.

    Unlike Llama/Mixtral SwiGLU there is no gate projection -- only ``up_proj``
    and ``down_proj``, with a squared-ReLU activation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(torchtitan.protocols.module.Module.Config):
        dim: int
        hidden_dim: int
        up_proj: Linear.Config | None = None
        down_proj: Linear.Config | None = None

        # No build() override here on purpose: Module.Config.build is what
        # attaches _param_init and _sharding_config to the built instance.
        # Overriding it skipped that, so this module's declared input
        # redistribution never ran -- under TP+SP the shared expert consumed
        # sequence-sharded input without the all-gather and returned half the
        # tokens ("size of tensor a (64) must match tensor b (32)"). The child
        # Linears still got their own configs, which is what made it look like
        # sharding was working.

    def __init__(self, config: Config):
        super().__init__()
        assert config.up_proj is not None and config.down_proj is not None
        self.up_proj = config.up_proj.build()
        self.down_proj = config.down_proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        return self.down_proj(torch.square(F.relu(h)))


class NemotronTransformerBlock(TransformerBlock):
    """
    Nemotron-3 Nano TransformerBlock: Hybrid Mamba-Transformer layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # One of "mamba" | "attention" | "mlp" | "moe". Nemotron-H layers hold a
        # SINGLE mixer each (pattern chars M, *, -, E respectively); attention and
        # the feed-forward are separate layers, never fused into one block.
        block_type: str = "mamba"
        is_mamba_block: bool = False
        mamba_num_heads: int = 128
        mamba_head_dim: int = 64
        # ``mamba_conv_dim`` is derived (see __init__); kept for backward compat.
        mamba_conv_dim: int = 8192
        mamba_n_groups: int = 8
        mamba_state_dim: int = 128
        mamba_conv_kernel: int = 4
        mamba_chunk_size: int = 256
        mamba_norm_eps: float = 1e-5
        # Timestep clamp. The reference uses (time_step_min, inf); NVIDIA's
        # released configs set time_step_min=0.001.
        mamba_dt_min: float = 0.001
        mamba_dt_max: float = float("inf")
        mamba_input_projection: Linear.Config | None = None
        mamba_output_projection: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        # block_type is authoritative; is_mamba_block is kept in sync for the
        # sharding/parallelize helpers that still key off it.
        self.block_type = config.block_type
        if config.is_mamba_block and config.block_type == "mamba":
            self.block_type = "mamba"
        self.is_mamba_block = self.block_type == "mamba"
        self.moe_enabled = self.block_type == "moe"
        
        self.mamba_num_heads = config.mamba_num_heads
        self.mamba_head_dim = config.mamba_head_dim
        self.mamba_n_groups = config.mamba_n_groups
        self.mamba_state_dim = config.mamba_state_dim
        self.chunk_size = config.mamba_chunk_size
        self.mamba_dt_min = config.mamba_dt_min
        self.mamba_dt_max = config.mamba_dt_max

        if self.is_mamba_block:
            self.norm = config.attention_norm.build()
            assert config.mamba_input_projection is not None
            dim = config.mamba_input_projection.in_features

            # Mamba-2 sizing, matching NemotronHMamba2Mixer:
            #   intermediate = num_heads * head_dim
            #   conv_dim     = intermediate + 2 * n_groups * state_dim   (x, B and C
            #                  are all short-convolved together)
            #   in_proj out  = intermediate (gate z) + conv_dim (x,B,C) + num_heads (dt)
            self.intermediate_size = self.mamba_num_heads * self.mamba_head_dim
            self.conv_dim = (
                self.intermediate_size + 2 * self.mamba_n_groups * self.mamba_state_dim
            )
            projection_size = self.intermediate_size + self.conv_dim + self.mamba_num_heads

            self.in_proj = nn.Linear(dim, projection_size, bias=False)
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=config.mamba_conv_kernel,
                groups=self.conv_dim,
                padding=config.mamba_conv_kernel - 1,
            )

            # Mamba-2 uses a per-head dt bias parameter, not a dt projection.
            self.dt_bias = nn.Parameter(torch.empty(self.mamba_num_heads, dtype=torch.float32))
            self.A_log = nn.Parameter(torch.empty(self.mamba_num_heads, dtype=torch.float32))
            self.D = nn.Parameter(torch.empty(self.mamba_num_heads, dtype=torch.float32))

            # Gated RMSNorm on the SSM output, grouped like the reference.
            self.mamba_norm = MambaRMSNormGated(
                self.intermediate_size,
                group_size=self.intermediate_size // self.mamba_n_groups,
                eps=config.mamba_norm_eps,
            )
            self.out_proj = nn.Linear(self.intermediate_size, dim, bias=False)
        elif self.block_type == "attention":
            assert config.attention is not None, "attention block requires an attention config"
            self.attention = config.attention.build()
            self.attention_norm = config.attention_norm.build()
        elif self.block_type == "mlp":
            assert config.feed_forward is not None, "mlp block requires a feed_forward config"
            assert config.ffn_norm is not None
            self.ffn_norm = config.ffn_norm.build()
            self.feed_forward = config.feed_forward.build()
        elif self.block_type == "moe":
            assert config.moe is not None, "moe block requires a moe config"
            assert config.ffn_norm is not None
            self.ffn_norm = config.ffn_norm.build()
            self.moe = config.moe.build()
        else:
            raise ValueError(
                f"Unknown block_type {self.block_type!r}; expected one of "
                "'mamba', 'attention', 'mlp', 'moe'."
            )

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

            # 1. Fused input projection -> (gate z, [x|B|C], dt)
            projected_states = self.in_proj(x_norm)
            z, x_bc, dt = torch.split(
                projected_states,
                [self.intermediate_size, self.conv_dim, self.mamba_num_heads],
                dim=-1,
            )

            # 2. Causal depthwise conv over x, B and C together, then SiLU.
            x_bc = x_bc.transpose(1, 2)
            x_bc = self.conv1d(x_bc)[:, :, :L]
            x_bc = x_bc.transpose(1, 2)
            x_bc = F.silu(x_bc)

            # 3. Split the convolved stream into x, B, C
            x_mamba, B_param, C_param = torch.split(
                x_bc,
                [
                    self.intermediate_size,
                    self.mamba_n_groups * self.mamba_state_dim,
                    self.mamba_n_groups * self.mamba_state_dim,
                ],
                dim=-1,
            )
            B_param = B_param.reshape(B, L, self.mamba_n_groups, self.mamba_state_dim)
            C_param = C_param.reshape(B, L, self.mamba_n_groups, self.mamba_state_dim)

            # 4. Per-head dt bias + softplus, then clamp the timestep.
            # The reference builds its limit as (time_step_min, inf) -- note it
            # uses time_step_min (0.001), NOT time_step_limit[0] (0.0) -- and
            # applies torch.clamp(dt, min=..., max=...). Omitting this only
            # shows up once real weights push dt below the floor, so it is easy
            # to miss with randomly-initialized parity tests.
            dt = F.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, min=self.mamba_dt_min, max=self.mamba_dt_max)

            # 5. Continuous A
            A = -torch.exp(self.A_log.float())

            x_mamba_reshaped = x_mamba.reshape(B, L, self.mamba_num_heads, self.mamba_head_dim)

            # 6. Mamba-2 chunked scan
            y = mamba2_chunk_scan(
                x_mamba_reshaped,
                dt,
                A,
                B_param,
                C_param,
                self.chunk_size,
                D=self.D,
            )

            y = y.reshape(B, L, -1).to(x_bc.dtype)

            # 7. Gated RMSNorm (gate applied inside), then output projection
            y = self.mamba_norm(y, z)
            out = self.out_proj(y)
            if is_flattened:
                out = out.squeeze(0)
            return x + out
        elif self.block_type == "attention":
            return x + self.attention(self.attention_norm(x), attention_masks, positions)
        elif self.block_type == "mlp":
            return x + self.feed_forward(self.ffn_norm(x))
        else:
            return x + self.moe(self.ffn_norm(x))

    def reset_parameters(self) -> None:
        """Initialize the raw-``nn`` Mamba parameters.

        torchtitan builds the model on the meta device and then calls
        ``to_empty()``, so anything not initialized here is left as uninitialized
        memory. ``Module.init_states`` only recurses into torchtitan ``Module``
        children, so the plain ``nn.Linear``/``nn.Conv1d``/``nn.Parameter``
        objects a Mamba block owns are initialized from this hook.
        """
        if not self.is_mamba_block:
            return

        # A spans a spectrum of timescales: A = -exp(A_log) = -(1..num_heads).
        # This init is POSITION-DEPENDENT, so unlike an elementwise fill it
        # cannot be applied to a local shard directly: under FSDP the parameter
        # is a DTensor whose local view is an arbitrary slice (often empty).
        # Build the full vector, then let distribute_tensor cut the matching
        # shard so every rank writes the right global slice.
        with torch.no_grad():
            A = torch.arange(
                1,
                self.mamba_num_heads + 1,
                dtype=self.A_log.dtype,
                device=self.A_log.device,
            )
            a_log_full = torch.log(A)
            if isinstance(self.A_log, DTensor):
                self.A_log.copy_(
                    distribute_tensor(
                        a_log_full, self.A_log.device_mesh, self.A_log.placements
                    )
                )
            else:
                self.A_log.copy_(a_log_full)

            # These two are constant fills, so they are shard-agnostic and work
            # on a DTensor as-is.
            nn.init.ones_(self.D)
            nn.init.ones_(self.dt_bias)

        for module in (self.in_proj, self.out_proj, self.conv1d):
            module.reset_parameters()
        self.mamba_norm.reset_parameters()

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

class Nemotron3Model(Decoder):
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

            from torchtitan.models.nemotron3.sharding import (
                set_nemotron_sharding_config,
            )

            set_nemotron_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            # get_nparams_and_active_nparams already weights routed-expert
            # params by the active ratio, so use its active count directly.
            # Deriving it here as nparams * top_k/num_experts was both wrong
            # (it scaled non-expert params too) and a ZeroDivisionError on the
            # dense flavors, which have num_experts == 0.
            nparams, active_nparams = get_nparams_and_active_nparams(model)

            # Only attention layers carry the quadratic term; in this hybrid
            # stack mamba/mlp/moe layers leave `attention` as None.
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                if attention is None:
                    continue
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops

    def verify_module_protocol(self) -> None:
        """Skip the plain-``nn.Module``-child check.

        Mamba blocks deliberately own raw ``nn.Linear``/``nn.Conv1d`` layers and
        bare ``nn.Parameter``s (in_proj, conv1d, dt_bias, A_log, D), which the
        default protocol check rejects. Those parameters are initialized by
        ``NemotronTransformerBlock.reset_parameters`` and distributed by its
        ``parallelize``; the guard below keeps that contract honest instead of
        letting the exemption hide a genuinely uninitialized parameter.
        """
        for name, module in self.named_modules():
            if isinstance(module, NemotronTransformerBlock) and module.is_mamba_block:
                if not hasattr(module, "reset_parameters"):
                    raise TypeError(
                        f"{name} is a Mamba block but has no reset_parameters(); "
                        "its raw nn parameters would be left uninitialized."
                    )
