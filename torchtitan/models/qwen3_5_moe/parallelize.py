# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for Qwen3.5 MoE hybrid decoder.

Handles the hybrid architecture with both full attention (Attention) and
linear attention (GatedDeltaNet) layers. Key design: maintain Shard(1)
residual stream throughout (SequenceParallel).

- Full attention layers: standard TP on wq/wk/wv/wo with SP on norms
- GatedDeltaNet layers: allgather input, Replicate DTensors internally,
  with DTensor-safe wrappers for conv1d (depthwise) and FLA kernel
- MoE: reuses apply_moe_ep_tp from llama4
- Shared expert: TP on w1/w3/w2 with Shard(1) residual
"""

import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

import torchtitan.models.qwen3_5_moe.model as _qwen3_model
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.dual_pipe_v import get_dual_pipe_v_flag
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.llama3.parallelize import apply_replicate
from torchtitan.models.llama4.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.models.qwen3_5_moe.model import Model
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn.default,
    torch._higher_order_ops.inductor_compiled_code,
}


# ---------------------------------------------------------------------------
# CP + TP helper — inner_attention needs plain tensors for CP dispatcher
# ---------------------------------------------------------------------------


class _DTensorSafeInnerAttention(nn.Module):
    """Wrapper that strips DTensor from Q/K/V before inner_attention and
    wraps the output back as DTensor.

    When TP produces DTensor Q/K/V (via ColwiseParallel with
    use_local_output=False) but CP's SDPA dispatcher expects plain tensors,
    this wrapper bridges the gap: it converts to local before the SDPA call
    and restores the DTensor placement after, so the subsequent gate
    multiplication (``output * sigmoid(gate)``) works with DTensor operands.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        is_dtensor = isinstance(q, DTensor)
        if is_dtensor:
            mesh, placements = q.device_mesh, q.placements
            q, k, v = q.to_local(), k.to_local(), v.to_local()

        out = self.inner(q, k, v, *args, **kwargs)

        if is_dtensor:
            out = DTensor.from_local(out, mesh, placements, run_check=False)
        return out


# ---------------------------------------------------------------------------
# GatedDeltaNet TP helpers — conv1d and FLA kernel need plain tensors
# ---------------------------------------------------------------------------


class _DTensorSafeConv1d(nn.Module):
    """Conv1d wrapper that bypasses DTensor dispatch for depthwise conv.

    DTensor's _tp_conv handler doesn't support depthwise conv (groups > 1).
    This wrapper stores weight as a Replicate DTensor (for mesh consistency
    needed by gradient norm clipping) but runs F.conv1d on local tensors.
    """

    def __init__(self, original: nn.Conv1d, tp_mesh: DeviceMesh):
        super().__init__()
        self.weight = nn.Parameter(
            DTensor.from_local(
                original.weight.data, tp_mesh, [Replicate()], run_check=False
            ),
            requires_grad=original.weight.requires_grad,
        )
        self.bias: nn.Parameter | None = None
        if original.bias is not None:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    original.bias.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=original.bias.requires_grad,
            )
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_dtensor = isinstance(x, DTensor)
        x_local = x.to_local() if is_dtensor else x
        w_local = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        b_local = None
        if self.bias is not None:
            b_local = (
                self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            )
        out = F.conv1d(
            x_local,
            w_local,
            b_local,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if is_dtensor:
            out = DTensor.from_local(out, x.device_mesh, x.placements, run_check=False)
        return out


_dispatch_patched = False
_softplus_registered = False


def _register_dtensor_softplus() -> None:
    """Register aten.softplus.default (and backward) as DTensor pointwise ops.

    PyTorch's DTensor pointwise op table includes silu, sigmoid, gelu, etc.
    but not softplus.  GatedDeltaNet uses F.softplus in its forward path,
    so we register it at runtime to avoid modifying PyTorch internals.
    """
    global _softplus_registered
    if _softplus_registered:
        return
    _softplus_registered = True

    from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
    from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy
    from torch.distributed.tensor._ops.registration import register_op_strategy

    register_op_strategy(
        torch.ops.aten.softplus.default,
        schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
    )(pointwise_strategy)

    register_op_strategy(
        torch.ops.aten.softplus_backward.default,
        schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
    )(pointwise_strategy)


def _install_dtensor_safe_dispatch() -> None:
    """Monkey-patch _gated_delta_rule_dispatch to handle DTensor inputs.

    The FLA CUDA kernel (and the torch_naive fallback) only accept plain
    tensors. This wrapper converts DTensor inputs to local before calling
    the original dispatch, then wraps the output back as DTensor.
    """
    global _dispatch_patched
    if _dispatch_patched:
        return
    _dispatch_patched = True

    original_dispatch = _qwen3_model._gated_delta_rule_dispatch

    def _dtensor_safe_dispatch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        if isinstance(q, DTensor):
            mesh, placements = q.device_mesh, q.placements
            out = original_dispatch(
                q.to_local(),
                k.to_local(),
                v.to_local(),
                g.to_local(),
                beta.to_local(),
                backend,
            )
            return DTensor.from_local(out, mesh, placements, run_check=False)
        return original_dispatch(q, k, v, g, beta, backend)

    _qwen3_model._gated_delta_rule_dispatch = _dtensor_safe_dispatch


def parallelize_qwen3_5_moe(
    model: Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    attn_backend = model.config.layer.attention.attn_backend
    if parallelism.context_parallel_degree > 1 and attn_backend not in (
        "sdpa",
        "varlen",
    ):
        raise NotImplementedError(
            f"Context Parallel only supports SDPA and varlen attention for Qwen3.5 MoE. "
            f"Got attn_backend='{attn_backend}'."
        )

    tp_mesh = None
    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            cp_enabled=parallel_dims.cp_enabled,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(
            parallelism=parallelism, ac_config=ac_config, parallel_dims=parallel_dims
        )

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
        )

    # Wrap inner_attention with DTensor-safe wrapper when needed.
    # varlen_attn doesn't support DTensor inputs (unlike SDPA which has
    # native DTensor dispatch), so we need this wrapper when TP is enabled.
    # CP also needs it for both SDPA and varlen backends.
    needs_dtensor_wrapper = (
        parallel_dims.tp_enabled and attn_backend == "varlen"
    ) or parallel_dims.cp_enabled
    if needs_dtensor_wrapper:
        # pyrefly: ignore [missing-attribute, not-callable]
        for block in model.layers.values():
            if block.layer_type == "full_attention":
                wrapper = _DTensorSafeInnerAttention(block.attn.inner_attention)
                block.attn.inner_attention = wrapper

    if parallel_dims.cp_enabled:
        # Apply CP to the actual attention module inside each wrapper
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [
                block.attn.inner_attention.inner
                for block in model.layers.values()
                if block.layer_type == "full_attention"
            ],
            parallel_dims.get_mesh("cp"),
            attn_backend,
        )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            # pyrefly: ignore [bad-argument-type]
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    cp_enabled: bool = False,
):
    """Apply tensor parallelism to non-MoE components.

    Handles the hybrid architecture:
    - Full attention layers: standard TP on Q/K/V/O projections
    - GatedDeltaNet layers: NoParallel on all submodules (Replicate DTensors)
      with DTensor-safe wrappers for conv1d and FLA kernel dispatch
    - Shared expert: TP on w1/w3/w2
    """
    # Patch FLA kernel dispatch to handle Replicate DTensor inputs (idempotent).
    _install_dtensor_safe_dispatch()
    # Register softplus as a DTensor pointwise op (not in PyTorch's default table).
    _register_dtensor_softplus()
    # Global: embedding, final norm, output head
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Per-layer plans
    positions_sharding = Replicate() if cp_enabled else None
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "ffn_norm": SequenceParallel(),
        }

        # pyrefly: ignore [missing-attribute]
        if transformer_block.layer_type == "full_attention":
            # Full attention: standard TP on Q/K/V/O projections
            layer_plan.update(
                {
                    "attn": PrepareModuleInput(
                        input_layouts=(Shard(1), Replicate(), None, positions_sharding),
                        desired_input_layouts=(
                            Replicate(),
                            Replicate(),
                            None,
                            positions_sharding,
                        ),
                    ),
                    "attn.wq": ColwiseParallel(use_local_output=False),
                    "attn.wk": ColwiseParallel(use_local_output=False),
                    "attn.wv": ColwiseParallel(use_local_output=False),
                    "attn.q_norm": SequenceParallel(sequence_dim=2),
                    "attn.k_norm": SequenceParallel(sequence_dim=2),
                    "attn.wo": RowwiseParallel(output_layouts=Shard(1)),
                }
            )
        else:
            # GatedDeltaNet: conv1d needs full sequence, FLA kernel needs
            # plain tensors.  Keep intermediates as Replicate DTensors so
            # that all params live on the TP mesh (required by gradient norm
            # clipping and optimizer), and only unwrap to local tensors at
            # the two incompatible operations (conv1d and FLA kernel).
            #
            # Pattern follows deepseek_v3 parallelize.py: NoParallel on
            # submodules that don't need TP but must be mesh-consistent.

            # Replace depthwise conv1d with DTensor-safe wrapper (must be
            # done before parallelize_module so the wrapper's weight is
            # already a Replicate DTensor).
            # pyrefly: ignore [missing-attribute]
            transformer_block.attn.conv1d = _DTensorSafeConv1d(
                transformer_block.attn.conv1d, tp_mesh
            )

            layer_plan.update(
                {
                    # Allgather input, reduce-scatter output at module
                    # boundary (same as before).
                    "attn": PrepareModuleInputOutput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        output_layouts=(Replicate(),),
                        desired_output_layouts=(Shard(1),),
                    ),
                    # All Linear submodules: params as Replicate DTensors,
                    # forward stays DTensor (use_local_output=False).
                    "attn.in_proj_qkv": NoParallel(use_local_output=False),
                    "attn.in_proj_z": NoParallel(use_local_output=False),
                    "attn.in_proj_a": NoParallel(use_local_output=False),
                    "attn.in_proj_b": NoParallel(use_local_output=False),
                    "attn.out_proj": NoParallel(use_local_output=False),
                    # Gated RMSNorm: weight as Replicate DTensor.
                    "attn.norm": NoParallel(use_local_output=False),
                }
            )

        # Shared expert gate + shared expert FFN
        layer_plan.update(
            {
                # shared_gate is nn.Linear(dim, 1) — gather input from
                # Shard(1) to Replicate, run Linear, slice output back
                # to Shard(1) (local only, no communication), then to
                # local tensor so it matches shared_ffn's local output.
                "shared_gate": NoParallel(
                    input_layout=Shard(1),
                    output_layout=Shard(1),
                    use_local_output=True,
                ),
                "shared_ffn": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "shared_ffn.w1": ColwiseParallel(),
                "shared_ffn.w2": RowwiseParallel(output_layouts=Shard(1)),
                "shared_ffn.w3": ColwiseParallel(),
            }
        )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

        # Distribute standalone GatedDeltaNet parameters (A_log, dt_bias)
        # as Replicate DTensors on the TP mesh.  These are nn.Parameters
        # directly on the module, not inside a submodule, so NoParallel
        # / parallelize_module cannot reach them.
        # pyrefly: ignore [missing-attribute]
        if transformer_block.layer_type != "full_attention":
            attn = transformer_block.attn
            attn.A_log = nn.Parameter(
                DTensor.from_local(
                    attn.A_log.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=attn.A_log.requires_grad,
            )
            attn.dt_bias = nn.Parameter(
                DTensor.from_local(
                    attn.dt_bias.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=attn.dt_bias.requires_grad,
            )

    logger.info("Applied Tensor Parallelism to the model")
