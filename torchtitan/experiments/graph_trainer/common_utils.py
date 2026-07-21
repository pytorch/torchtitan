# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate
from torch.fx.traceback import annotate_fn
from torch.utils._pytree import register_constant, register_pytree_node, tree_map

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.tools.logging import logger


BOXED_CODEGEN_META = "graph_trainer_boxed_codegen"
LossResult: TypeAlias = torch.Tensor | tuple[torch.Tensor, dict[str, Any]]
AnnotatedLossFn: TypeAlias = Callable[..., LossResult]


@contextmanager
def log_timer(label: str):
    start = time.perf_counter()
    yield
    elapsed_s = time.perf_counter() - start
    logger.info("%s took %.3fs", label, elapsed_s)


def build_decoder_config_for_backend(
    config_builder: Callable, attn_backend: str, **builder_kwargs
):
    """Build a Decoder model config for ``attn_backend``, allowing test-only SDPA.

    ``SDPA`` is not a valid production language-model backend — ``get_attention_config``
    rejects it because the dataloaders always emit per-document positions and SDPA
    cannot consume them (it only has a boolean ``is_causal``). The graph_trainer
    tests, however, use SDPA to exercise *backend-agnostic* graph machinery
    (precompile-artifact serialization, custom codegen, context parallel, bitwise
    determinism) without FlexAttention's ``BlockMask``, which is unpicklable (its
    ``mask_mod`` closures are Python code objects), is not a tensor (so it breaks
    pipeline-parallel split-backward, which calls ``.requires_grad`` on every stage
    input), and overflows the fp32 Triton shared-memory limit on large head dims.

    For SDPA we build the flex config (a valid backend) and swap each layer's
    ``inner_attention`` to ``ScaledDotProductAttention.Config()``. Production code
    never reaches this path: ``get_attention_config`` still rejects ``sdpa``, so no
    model registry can construct an SDPA language model outside these tests.
    """
    if attn_backend != "sdpa":
        return config_builder(attn_backend=attn_backend, **builder_kwargs)

    from torchtitan.models.common.attention import ScaledDotProductAttention

    config = config_builder(attn_backend="flex", **builder_kwargs)
    for layer in config.layers:
        layer.attention.inner_attention = ScaledDotProductAttention.Config()
    return config


def _local_stride(tensor: torch.Tensor) -> tuple[int, ...]:
    return (
        tensor.to_local().stride() if isinstance(tensor, DTensor) else tensor.stride()
    )


def _maybe_materialize_grad_for_param_layout(
    param: torch.Tensor, grad: torch.Tensor
) -> torch.Tensor:
    """Match eager autograd's ``param.grad`` layout contract after graph replay.

    Graph replay assigns ``torch.autograd.grad`` outputs manually, bypassing
    AccumulateGrad's normal stride materialization. Copying through
    ``empty_like(param)`` restores the param's global and DTensor-local layout
    when Inductor returns an equivalent but differently-strided grad.
    """
    if grad.stride() == param.stride() and _local_stride(grad) == _local_stride(param):
        return grad

    materialized_grad = torch.empty_like(param)
    materialized_grad.copy_(grad)
    return materialized_grad


def set_graph_module_boxed_codegen(
    gm: torch.fx.GraphModule,
    *,
    boxed: bool,
) -> torch.fx.GraphModule:
    """Set the FX calling convention and record it in graph metadata.

    Args:
        gm: Graph module whose generated Python wrapper should be updated.
        boxed: Whether the wrapper should accept one mutable argument list and
            clear it after placeholder extraction.

    Returns:
        The same graph module after recompilation if the calling convention
        changed.
    """

    if gm.meta.get(BOXED_CODEGEN_META) is boxed:
        return gm
    codegen = torch.fx.graph._BoxedCodeGen() if boxed else torch.fx.graph.CodeGen()
    gm.graph.set_codegen(codegen)
    gm.recompile()
    gm.meta[BOXED_CODEGEN_META] = boxed
    return gm


def ensure_boxed_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Use boxed FX codegen for runtime-owned graph modules.

    Args:
        gm: Graph module to box.

    Returns:
        The same graph module with boxed FX codegen enabled.
    """

    return set_graph_module_boxed_codegen(gm, boxed=True)


_MODULE_FQN = "module_fqn"
_EP_TOKEN_COUNT_EXCHANGE = "EP_token_count_exchange"
_EP_TOKEN_COUNT_SYNC = "EP_token_count_sync"
_EP_TOKEN_EXCHANGE = "EP_token_exchange"
_EP_TOKEN_EXCHANGE_WAIT = "EP_token_exchange_wait"
_NOT_IN_LAYERS = -1


def compute_annotated_loss(
    loss_fn: AnnotatedLossFn,
    pred: torch.Tensor,
    labels: torch.Tensor,
    loss_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute the loss tensor with the same FX metadata convention as GraphTrainer."""
    annotated_loss_fn = annotate_fn({_MODULE_FQN: "loss"})(loss_fn)
    result = annotated_loss_fn(pred, labels, **(loss_kwargs or {}))
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "GraphTrainer loss functions must return a loss tensor or "
                "(loss tensor, metrics)."
            )
        loss, _metrics = result
        return loss
    return result


def accumulate_param_grads_(
    params: Iterable[torch.Tensor],
    grads: Iterable[torch.Tensor | None],
) -> None:
    """Accumulate explicit graph-produced gradients into live parameters."""
    for param, grad in zip(params, grads, strict=True):
        if grad is None:
            continue
        grad = _maybe_materialize_grad_for_param_layout(param, grad)
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad


def _is_backward_node(node: torch.fx.Node) -> bool:
    return node.meta.get("autograd_backward", False)


def _get_module_fqn(node: torch.fx.Node) -> str:
    return node.meta.get("custom", {}).get(_MODULE_FQN, "")


def _get_layer_id(node: torch.fx.Node) -> int:
    """Extract the layer index from the node's module_fqn metadata.

    Nodes under ``layers.<N>`` return ``N``.
    All other nodes (tok_embeddings, norm, output) return ``_NOT_IN_LAYERS``.
    """
    fqn = _get_module_fqn(node)
    parts = fqn.split(".")
    if parts[0] == "layers" and len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return _NOT_IN_LAYERS


def annotate_module_fqns(model: nn.Module) -> None:
    """Annotate all modules' forward with their fully-qualified names.

    Every named submodule (excluding the root) gets its forward method wrapped
    with ``annotate_fn`` so that FX nodes carry ``module_fqn`` in
    ``node.meta["custom"]``.

    Call once after model construction, before tracing/compilation.
    """
    for fqn, submodule in model.named_modules():
        if fqn:  # skip root module
            submodule.forward = annotate_fn({_MODULE_FQN: fqn})(submodule.forward)


def matches_module_fqn_pattern(pattern: str, fqn: str) -> bool:
    """Match one module FQN against a component-wise fnmatch pattern."""
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    return len(pattern_parts) == len(fqn_parts) and all(
        fnmatch.fnmatchcase(fqn_part, pattern_part)
        for pattern_part, fqn_part in zip(pattern_parts, fqn_parts)
    )


_MOE_EP_REGIONS_ANNOTATED = False


def annotate_moe_ep_regions() -> None:
    """Annotate MoE EP compute, dispatch, and combine regions for FX passes."""
    global _MOE_EP_REGIONS_ANNOTATED
    if _MOE_EP_REGIONS_ANNOTATED:
        return

    from torchtitan.models.common.moe import MoE
    from torchtitan.models.common.token_dispatcher import (
        AllToAllTokenDispatcher,
        LocalTokenDispatcher,
    )

    LocalTokenDispatcher.dispatch = annotate_fn({"EP": "dispatch"})(
        LocalTokenDispatcher.dispatch
    )
    LocalTokenDispatcher.combine = annotate_fn({"EP": "combine"})(
        LocalTokenDispatcher.combine
    )
    AllToAllTokenDispatcher.dispatch = annotate_fn({"EP": "dispatch"})(
        AllToAllTokenDispatcher.dispatch
    )
    AllToAllTokenDispatcher.combine = annotate_fn({"EP": "combine"})(
        AllToAllTokenDispatcher.combine
    )
    AllToAllTokenDispatcher._token_count_exchange = annotate_fn(
        {_EP_TOKEN_COUNT_EXCHANGE: "dispatch"}
    )(AllToAllTokenDispatcher._token_count_exchange)
    AllToAllTokenDispatcher._sync_token_count_exchange = annotate_fn(
        {_EP_TOKEN_COUNT_SYNC: "dispatch"}
    )(AllToAllTokenDispatcher._sync_token_count_exchange)
    AllToAllTokenDispatcher._dispatch_token_exchange = annotate_fn(
        {_EP_TOKEN_EXCHANGE: "dispatch"}
    )(AllToAllTokenDispatcher._dispatch_token_exchange)
    AllToAllTokenDispatcher._combine_token_exchange = annotate_fn(
        {_EP_TOKEN_EXCHANGE: "combine"}
    )(AllToAllTokenDispatcher._combine_token_exchange)
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)
    _MOE_EP_REGIONS_ANNOTATED = True


def parallelize_inputs(parallel_dims, args, kwargs):
    if not parallel_dims.tp_enabled:
        return args, kwargs

    def to_dtensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return DTensor.from_local(
                tensor, parallel_dims.get_mesh("tp"), [Replicate()]
            )
        return tensor

    dt_args = tree_map(to_dtensor, args)

    # TODO: When using flex_attention, BlockMask would show up in kwargs,
    # and it's unclear how to convert it to DTensor. If I use to_dtensor,
    # it would fail with Dynamo Error: P2011360347
    # dt_kwargs = tree_map(to_dtensor, kwargs)

    dt_kwargs = kwargs

    return dt_args, dt_kwargs


def register_blockmask_pytree_node():
    from torch.nn.attention.flex_attention import BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_pytree_node(
            BlockMask,
            BlockMask._flatten,
            BlockMask._unflatten,
            flatten_with_keys_fn=BlockMask._flatten_with_keys,
            serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
        )


def maybe_register_blockmask_pytree_node() -> None:
    """Register flex-attention pytree helpers if they are missing."""
    from torch.nn.attention.flex_attention import _MaskModWrapper, BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_blockmask_pytree_node()
    if _MaskModWrapper not in torch.utils._pytree.SUPPORTED_NODES:
        register_constant(_MaskModWrapper)


def end_with_pass(passes: list[Callable], names: list[str]) -> bool:
    return (
        len(passes) > 0
        and (last_pass_name := getattr(passes[-1], "__name__", None))
        and (last_pass_name in names)
    )


def get_default_transformer_block_buckets(
    n_layers: int,
    *,
    chunked_loss_enabled: bool = False,
    moe_layer_ids: frozenset[int] = frozenset(),
    split_moe_expert_buckets: bool = False,
) -> list[list[str] | str]:
    """Get default transformer block buckets for manual bucketing passes.

    Assumes the standard Decoder layout: tok_embeddings, layers.0..N-1,
    norm, and output (e.g., Llama3, DeepSeekV3, Qwen3).
    """
    layer_buckets: list[list[str] | str] = []
    for layer_id in range(n_layers):
        if layer_id in moe_layer_ids and split_moe_expert_buckets:
            layer_buckets.extend(
                [
                    [
                        f"layers.{layer_id}.attention_norm",
                        f"layers.{layer_id}.attention",
                        f"layers.{layer_id}.ffn_norm",
                        f"layers.{layer_id}.moe.router",
                        f"layers.{layer_id}.moe.shared_experts",
                    ],
                    f"layers.{layer_id}.moe.routed_experts.inner_experts",
                ]
            )
        else:
            layer_buckets.append(f"layers.{layer_id}")
    final_bucket = ["norm", "lm_head"]
    if chunked_loss_enabled:
        # Chunked loss moves the lm_head weight use under module_fqn "loss".
        final_bucket.append("loss")

    return [
        "tok_embeddings",
        *layer_buckets,
        final_bucket,
    ]


def get_transformer_block_buckets(model) -> list[list[str] | str]:
    """Get transformer block buckets for manual bucketing passes.

    Works for any model with tok_embeddings, layers (OrderedDict), norm, and output
    attributes (e.g., Llama3, DeepSeekV3).
    """
    # [TODO](ruisizhang123) add EP support for transformer block bucketing
    module_list = [
        model.tok_embeddings,
        [model.norm, model.lm_head],
    ]
    for layer_id, transformer_block in model.layers.items():
        module_list.append(transformer_block)

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                if fqn_list := convert_modules_to_fqns(m, module_to_fqn_mapping):
                    result.append(fqn_list)
            else:
                if fqn := module_to_fqn_mapping.get(m):
                    result.append(fqn)
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def apply_cp_to_attention(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Wrap each layer's inner attention with CP logic."""
    attention_modules = [
        # pyrefly: ignore [missing-attribute]
        block.attention.inner_attention
        for block in model.layers.values()
    ]
    apply_cp_to_forward(attention_modules, parallel_dims.get_mesh("cp"))


def apply_simple_fsdp(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> nn.Module:
    """Wrap the model (and any MoE experts) with graph_trainer's simple_fsdp.

    For MoE-enabled models, the ``moe.routed_experts.inner_experts`` submodules
    (the routed-expert weights) are separately wrapped on the EDP mesh when expert
    parallelism is enabled.
    """
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            dp_mesh_dim_names = ["dp_replicate", "fsdp"]
            dp_mode = "hybrid_shard"
        else:
            dp_mesh_dim_names = ["dp_replicate"]
            dp_mode = "replicate"
    else:
        dp_mesh_dim_names = ["fsdp"]
        dp_mode = "fully_shard"

    dp_mesh = parallel_dims.get_mesh(dp_mesh_dim_names)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    if parallel_dims.ep_enabled and isinstance(model, Decoder):
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        assert edp_mesh is not None

        for _, transformer_block in model.layers.items():
            if not isinstance(transformer_block, TransformerBlock):
                continue
            moe = getattr(transformer_block, "moe", None)
            if moe is None:
                continue
            inner_experts = moe.routed_experts.inner_experts
            experts_shard_dim = 0
            if edp_mesh["efsdp"].size() * parallel_dims.ep > inner_experts.num_experts:
                experts_shard_dim = 1

            moe.routed_experts.inner_experts = data_parallel(
                inner_experts,
                edp_mesh,
                dp_mode,
                mp_policy=mp_policy,
                shard_dim=experts_shard_dim,
            )

    model = data_parallel(
        model,
        dp_mesh,
        dp_mode,
        mp_policy=mp_policy,
    )
    logger.info(
        "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
    )
    return model
