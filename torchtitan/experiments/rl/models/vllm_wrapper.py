# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base wrapper for TorchTitan models to work with vLLM V1 engine.

This module provides TorchTitanVLLMModel: Core model class that adapts
TorchTitan models for vLLM.
"""

import dataclasses
from dataclasses import dataclass
from functools import partial

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    apply_overrides,
    CompileConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.experiments.rl.models.attention import VLLMAttentionWrapper
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.utils import torch_utils as _torch_utils


logger = init_logger(__name__)

# NOTE: Monkeypatch vLLM's weak_ref_tensor to handle DTensor
# This is because piecewise CUDA-graph capture calls weak_ref_tensor()
# on every subgraphoutput (see vllm/compilation/cuda_graph.py).
# When TP is active some of those outputs are DTensors which fail with
# ("The specified pointer resides on host memory").  to_local
# converts the DTensor to a plain tensor. which succeeds with this
# cudagraph implementation.
_original_weak_ref_tensor = _torch_utils.weak_ref_tensor


def _dtensor_safe_weak_ref_tensor(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return _original_weak_ref_tensor(tensor)


_torch_utils.weak_ref_tensor = _dtensor_safe_weak_ref_tensor


# Process-global: install the op swap at most once even if the generator is
# re-initialized in the same process (re-wrapping would chain shims).
_tp_all_reduce_patched = False


def _patch_vllm_all_reduce() -> None:
    """Route the generator's tensor-parallel all-reduce through vLLM's custom
    one-shot/multimem AR instead of DTensor's NCCL ring redistribute (applied
    when batch-invariant mode is off). Idempotent. Three changes (see numbered
    comments below); changes 1-2 cover the default (DTensor) backend, change 3
    additionally routes the spmd_types backend's redistribute through the same op:

    1. Swap torch.ops._c10d_functional.all_reduce (the op every DTensor
       Partial -> Replicate redistribute calls) for
       tensor_model_parallel_all_reduce, which reuses vLLM's TP GroupCoordinator
       -- the same ranks, but the group that owns the custom-AR shared buffers.
       DTensor wraps the synchronous result in an AsyncCollectiveTensor whose
       wait_tensor is a no-op. Only sum reductions are routed (the custom AR is
       sum-only); others fall through to the original op. No-op at world_size 1.

    2. Force the custom AR onto its registered=False path so cudagraph capture
       works. registered=True records graph buffers and calls cudaIpcGetMemHandle
       on them, which fails for the expandable_segments (VMM) memory the RL stack
       enables for Monarch RDMA. registered=False reduces via the init-time
       buffer_ptrs (raw cudaMalloc, IPC-able), records no graph buffers, at the
       cost of one staging copy per AR.

    TODO: this is a stopgap to close the generator's TP all-reduce perf gap.
    Improve our native (DTensor) all-reduce path and remove this patch.
    """
    global _tp_all_reduce_patched
    if _tp_all_reduce_patched:
        return

    from vllm.distributed import get_tp_group

    c10d = torch.ops._c10d_functional
    # Op schema: all_reduce(Tensor input, str reduce_op, Any group_name) -> Tensor.
    original_all_reduce = c10d.all_reduce

    def all_reduce(input, reduce_op, group_name):
        if reduce_op == "sum":
            # Use vLLM's helper rather than the incoming group_name: the custom-AR
            # kernel is bound to vLLM's TP GroupCoordinator and the shared buffers
            # it registered, so the reduction must run on vLLM's TP group, not
            # DTensor's TP mesh PG (a different PG over the same ranks -> rank-for-
            # rank equivalent). The helper resolves that group, guards
            # world_size==1, and on CUDA dispatches to torch.ops.vllm.all_reduce.
            return tensor_model_parallel_all_reduce(input)
        return original_all_reduce(input, reduce_op, group_name)

    c10d.all_reduce = all_reduce

    # Force vLLM's TP custom AR onto its registered=False path (see point 2).
    device_comm = get_tp_group().device_communicator
    ca = getattr(device_comm, "ca_comm", None) if device_comm is not None else None
    if ca is not None and not ca.disabled:

        def custom_all_reduce(input):
            # Mirrors CustomAllreduce.custom_all_reduce but always registered=False.
            if ca.disabled or not ca.should_custom_ar(input):
                return None
            return ca.all_reduce(input, registered=False)

        ca.custom_all_reduce = custom_all_reduce

    # 3. (spmd_types) spmd's redistribute issues an in-place dist.all_reduce that
    #    step 1 can't see, so redirect its Partial->{R,I} reduce to the functional
    #    collective. Guarded on no-grad: funcol's backward is wrong for this reduce,
    #    so only redirect during inference (no backward); training is unaffected.
    import torch.distributed._functional_collectives as funcol

    original_redistribute = spmd.redistribute

    def redistribute(x, group, *, src, dst, **kwargs):
        if src == spmd.P and dst in (spmd.R, spmd.I) and not torch.is_grad_enabled():
            return funcol.wait_tensor(funcol.all_reduce(x, reduceOp="sum", group=group))
        return original_redistribute(x, group, src=src, dst=dst, **kwargs)

    spmd.redistribute = redistribute

    _tp_all_reduce_patched = True
    logger.info(
        "vllm_allreduce: routed _c10d_functional.all_reduce (TP sum reductions) "
        "through vLLM custom all-reduce (registered=False, cudagraph-safe)"
    )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
    }
)
class VLLMModelWrapper(Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models. Implemented
    required interface required by vLLM Engine.
    Doc: https://docs.vllm.ai/en/latest/contributing/model/basic/
    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Parallelism setup and DTensor conversion between torchtitan and vLLM
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        parallelism: InferenceParallelismConfig,
        compile_config: CompileConfig,
        checkpoint_config: CheckpointManager.Config,
        vllm_config: VllmConfig,
        prefix: str = "",
        override: OverrideConfig,
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components from model_spec
        self.state_dict_adapter = model_spec.state_dict_adapter
        self.parallelize_fn = model_spec.parallelize_fn

        # Replace inner_attention with VLLMAttentionWrapper in config
        model_config = model_spec.model
        attn_config = model_config.layers[0].attention
        n_heads = attn_config.n_heads
        n_kv_heads = attn_config.n_kv_heads or n_heads
        head_dim = (
            attn_config.head_dim
            if attn_config.head_dim is not None
            else model_config.dim // n_heads
        )
        new_layers = []
        for layer_cfg in model_config.layers:
            vllm_backend = VLLMAttentionWrapper.Config(
                hidden_size=model_config.dim,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                head_dim=head_dim,
                sliding_window_size=getattr(
                    layer_cfg.attention, "sliding_window_size", None
                ),
            )
            new_layers.append(
                dataclasses.replace(
                    layer_cfg,
                    attention=dataclasses.replace(
                        layer_cfg.attention, inner_attention=vllm_backend
                    ),
                )
            )
        self.config = dataclasses.replace(model_config, layers=new_layers)
        logger.debug(f"Creating model with config: {self.config.to_dict()}")

        # Translate the inference parallelism into torchtitan's full
        # ParallelismConfig that ParallelDims / parallelize_fn consume.
        training_parallelism = parallelism.to_training()

        # Build ParallelDims from the translated ParallelismConfig so TP/EP
        # sharding sees the same mesh shape as vLLM. data_parallel_shard_degree
        # carries vLLM's pure DP here (skip_dp=True below), not TorchTitan FSDP.
        self.parallel_dims = ParallelDims(
            dp_replicate=training_parallelism.data_parallel_replicate_degree,
            dp_shard=training_parallelism.data_parallel_shard_degree,
            cp=training_parallelism.context_parallel_degree,
            tp=training_parallelism.tensor_parallel_degree,
            pp=training_parallelism.pipeline_parallel_degree,
            ep=training_parallelism.expert_parallel_degree,
            world_size=dist.get_world_size(),
            spmd_backend=training_parallelism.spmd_backend,
        )
        dist_utils.set_spmd_backend(training_parallelism.spmd_backend)
        self.spmd_context = dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=False,
        )

        # Fill sharding configs on the config BEFORE build so every sub-module
        # is constructed with its ShardingConfig attached (required by the
        # declarative model.parallelize() API). Need to be called after Attention
        # module replacement.
        # Provides the generic config shape (has .parallelism) so
        # update_from_config can extract parallelism uniformly.
        @dataclass(kw_only=True, slots=True)
        class _InferenceConfig:
            parallelism: ParallelismConfig

        self.config.update_from_config(
            config=_InferenceConfig(parallelism=training_parallelism)
        )

        # Apply config overrides (e.g. the fused gate+up SwiGLU) after
        # update_from_config (which fills the sharding the override factories
        # read) and before build
        if override.imports:
            apply_overrides(override, self.config)

        # Build model on meta device to avoid allocating full model on every GPU
        with torch.device("meta"):
            self.model = self.config.build()

        self.model = self.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            training=TrainingConfig(),
            parallelism=training_parallelism,
            compile_config=compile_config,
            ac_config=None,
            dump_folder="",
            # Generator inference replicates parameters across vLLM DP groups.
            # Keep TP/EP sharding above, but do not translate dp_shard into
            # TorchTitan FSDP/DDP here.
            skip_dp=True,
        )

        # Load initial weights based on checkpoint config.
        self._checkpoint_config = checkpoint_config

        # Materialize model on GPU — only allocates local shards (not full
        # model) thanks to EP/TP DTensor sharding applied above.
        self.model.to_empty(device=vllm_config.device_config.device)
        # HF checkpoints do not necessarily contain every TorchTitan buffer
        # (for example MoE expert_bias_E).
        # TODO: When checkpoint doesn't contains expert_bias_E, check the config
        # should use loss based load balancing strategy.
        with torch.no_grad():
            # spmd_types parameter init needs the current mesh to materialize
            # local shards for fused parameters, including the fused QKV linear
            # used by model variants such as Qwen3.
            # TODO: Consider an init_non_persistent_buffers contract on the
            # Decoder / Model class so buffer-only init does not need this
            # spmd context.
            with self.spmd_context():
                self.model.init_weights(buffer_device=None)
        self._maybe_initial_load_weights()

        # Give each gpt-oss attention's vLLM backend its sink rescale.
        # Need to do it here after parallelize + weight load so sinks are
        # TP-sharded.
        self._inject_attention_sinks()

        # Route the TP all-reduce through vLLM's custom AR (off under
        # batch-invariant mode, where its size-dependent algorithm breaks).
        if self.parallel_dims.tp_enabled and not is_in_batch_invariant_mode():
            _patch_vllm_all_reduce()

    # TODO: followup with potentially adding extra kwarg ``sinks`` to vLLM attn
    def _inject_attention_sinks(self) -> None:
        """Give each gpt-oss attention's vLLM backend its sink-rescale hook."""
        from torchtitan.models.gpt_oss.model import (
            apply_attention_sink_rescale,
            Attention,
        )

        for module in self.model.modules():
            if not isinstance(module, Attention):
                continue
            sinks = module.sinks
            local_sinks = sinks._local_tensor if isinstance(sinks, DTensor) else sinks
            module.inner_attention.vllm_attn.impl.out_transform = partial(
                apply_attention_sink_rescale, sinks=local_sinks
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings."""
        with self.spmd_context():
            return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        vLLM required API.
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs [total_tokens] (1D varlen format)
            positions: Position indices [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        with self.spmd_context():
            # Convert vLLM interface to TorchTitan interface
            # vLLM: [total_tokens] -> TorchTitan: [batch_size, seq_len]
            tokens_2d = input_ids.unsqueeze(0)

            # Get embeddings
            h = self.model.tok_embeddings(tokens_2d)

            positions = positions.unsqueeze(0)

            # Pass through transformer layers
            for layer in self.model.layers.values():
                h = layer(h, attention_masks=None, positions=positions)

            h = self.model.norm(h)
        # Inference disables sequence parallelism, so final hidden states should
        # already be replicated before returning to vLLM.
        if isinstance(h, DTensor):
            assert all(isinstance(p, Replicate) for p in h.placements)
            h = h._local_tensor

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            hidden_size = h.size(-1)
            h = h.view(-1, hidden_size)
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """vLLM required API.
        Compute logits from hidden states."""

        with self.spmd_context():
            # When TP is applied, forward() returns the full tensor back to vLLM.
            # The DTensor path wraps that plain tensor before lm_head; spmd_types
            # keeps tensors local and uses the module sharding contracts directly.
            if (
                self.parallel_dims.tp_enabled
                and self.parallel_dims.spmd_backend != "spmd_types"
            ):
                hidden_states = DTensor.from_local(
                    hidden_states,
                    device_mesh=self.parallel_dims.get_mesh("tp"),
                    placements=[
                        Replicate(),
                    ],
                )

            logits = self.model.lm_head(hidden_states)

            # lm_head returns vocab-sharded logits under TP; gather to the
            # full local logits tensor that vLLM expects.
            if self.parallel_dims.tp_enabled:
                if self.parallel_dims.spmd_backend == "spmd_types":
                    mesh = current_spmd_mesh()
                    assert mesh is not None
                    logits = spmd.redistribute(
                        logits,
                        mesh.get_group("tp"),
                        src=spmd.S(-1),
                        dst=spmd.R,
                        backward_options={"op_dtype": logits.dtype},
                    )
                elif isinstance(logits, DTensor):
                    placements = tuple(
                        Replicate()
                        if isinstance(p, Shard) and p.dim in (-1, logits.ndim - 1)
                        else p
                        for p in logits.placements
                    )
                    logits = logits.redistribute(placements=placements).to_local()

        return logits

    def _maybe_initial_load_weights(self) -> None:
        """Load initial HF weights via CheckpointManager.

        Controlled by ``self._checkpoint_config``:
        - ``enable=True`` and ``initial_load_in_hf=True``: load from HF
          via CheckpointManager (standalone inference path).
        - ``enable=False``: skip (RL loop — weights arrive via TorchStore).
        """
        cfg = self._checkpoint_config
        if not cfg.enable:
            return

        sd_adapter = None
        if self.state_dict_adapter is not None:
            sd_adapter = self.state_dict_adapter(
                model_config=self.config,
                hf_assets_path=cfg.initial_load_path,
            )

        # Model-only CheckpointManager: initial_load_model_only=True (default)
        # ensures only MODEL state is loaded, so None optimizer/lr_scheduler
        # are never accessed.
        checkpointer = cfg.build(
            dataloader=None,
            model_parts=[self.model],
            optimizers=None,
            lr_schedulers=None,
            states={},
            sd_adapter=sd_adapter,
        )
        checkpointer.load()

    def load_weights(self, weights_iter):
        """
        vLLM required API.

        This is a no-op method since model weights are already loaded during initialization.
        Returns the names of all parameters that have been loaded so vLLM's safety check passes.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """

        loaded_param_names = set()
        for name, _ in self.model.named_parameters():
            loaded_param_names.add("model." + name)

        logger.info(
            f"Weights already loaded during model initialization. \
            Returning {len(loaded_param_names)} loaded parameter names to satisfy vLLM safety check."
        )

        # Return the names of all loaded parameters so vLLM knows they were handled
        return loaded_param_names
