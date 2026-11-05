# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-wide memory and pipeline runtime for standalone Dist-MoE experts.

With ``Config.inference`` set, the runtime serves a vLLM generator, which runs no
backward and no pipeline: it plans scratch memory only, and each call carries
only the rows its step needs (see ``set_num_local_input_tokens_per_call``).

Shape suffixes in this file use ``T`` for local input tokens, ``K`` for selected
experts, ``E`` for local experts, ``F`` for the expert intermediate dimension,
and ``D`` for the model dimension.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torch
from torch.distributed.pipelining import (
    analyze_pipeline_activation_liveness,
    PipelineStageInfo,
)
from torch.distributed.pipelining.schedules import PipelineScheduleMulti
from torch.utils.hooks import RemovableHandle

from torchtitan.config import Configurable
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode

from . import _dist_moe as dist_moe


if TYPE_CHECKING:
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.common.dist_moe.routed_experts import DistMoeRoutedExperts


logger = logging.getLogger(__name__)

__all__ = ["DistMoeRuntime"]

PPActivationSlotPolicy = Literal["stage_microbatch", "microbatch"]

# Inference picks its BF16 expert-GEMM schedule from the call size, and some
# schedules round a row differently depending on the call size or on the other
# rows of its expert group. Batch-invariant mode fixes the schedule to the tile
# shape of the trainer's fixed schedule.
_BATCH_INVARIANT_BF16_GROUPED_GEMM_PRESET = "2cta2mma_bm512_bn256"


@dataclass(frozen=True, slots=True)
class _DistMoePipelineActivationPlan:
    """Static activation-slot assignment for Dist-MoE stages on one PP rank.

    PyTorch's schedule analysis colors the live interval from each stage forward
    through the backward action that releases its saved state. The resulting
    slot IDs are reused only for non-overlapping intervals. Dist-MoE additionally
    records the maximum MoE-layer depth supported by every slot because the
    annex planner stores all those layers in the same slot.

    Attributes:
        max_live_activation_slots: Number of physical slots required by the
            schedule.
        max_moe_layers_per_activation_slot: Maximum local MoE-layer depth that
            can append saved state to one slot.
        activation_slot_id_by_stage_and_microbatch: Physical slot ID for every
            global stage and microbatch pair.
    """

    max_live_activation_slots: int
    max_moe_layers_per_activation_slot: int
    activation_slot_id_by_stage_and_microbatch: dict[tuple[int, int], int]


class _DistMoeForwardContext:
    """Resolve one schedule-colored activation slot for a stage forward.

    Eager PP enters this object as a stage forward context, which updates the
    annex's stable device scalar before model execution. GraphPP asks the same
    object for an immutable one-element view and supplies that view as an
    explicit stage-graph input. The coloring policy and slot ownership are
    therefore shared without making GraphPP depend on eager stage hooks.
    """

    def __init__(
        self,
        context: dist_moe.Context,
        *,
        active_stage_indices: frozenset[int],
        activation_slot_id_by_stage_and_microbatch: dict[tuple[int, int], int],
        activation_slot_ids_S: torch.Tensor,
        max_moe_layers_per_activation_slot: int,
    ) -> None:
        self._context = context
        self._active_stage_indices = active_stage_indices
        self._activation_slot_id_by_stage_and_microbatch = (
            activation_slot_id_by_stage_and_microbatch
        )
        self._activation_slot_ids_S = activation_slot_ids_S
        self._max_moe_layers_per_activation_slot = max_moe_layers_per_activation_slot

    def _slot_id(self, info: PipelineStageInfo) -> int | None:
        if info.stage_index not in self._active_stage_indices:
            return None
        if not self._activation_slot_id_by_stage_and_microbatch:
            return 0
        key = (info.stage_index, info.microbatch_index)
        try:
            return self._activation_slot_id_by_stage_and_microbatch[key]
        except KeyError as error:
            raise ValueError(
                "Dist-MoE has no activation-slot assignment for "
                f"stage {info.stage_index}, microbatch {info.microbatch_index}"
            ) from error

    def resolve_activation_slot(
        self,
        info: PipelineStageInfo,
    ) -> torch.Tensor | None:
        """Return the immutable one-element slot view for a graph forward."""
        slot_id = self._slot_id(info)
        if slot_id is None:
            return None
        return self._activation_slot_ids_S.narrow(0, slot_id, 1)

    @contextmanager
    def __call__(self, info: PipelineStageInfo) -> Iterator[None]:
        """Select the slot used by one eager pipeline forward."""
        slot_id = self._slot_id(info)
        if slot_id is not None:
            self._context.select_activation_slot(
                slot_id,
                self._max_moe_layers_per_activation_slot,
            )
        yield


class DistMoeRuntime(Configurable):
    """Own one annex context shared by all local Dist-MoE expert modules.

    Forward/backward initialization prepares the runtime after model
    parallelization because its memory plan depends on the final local stages,
    expert-parallel process group, and PP schedule. The annex resolves WGrad
    storage from each live parameter's declared gradient dtype or existing
    gradient storage during eager execution. GraphTrainer supplies the BF16
    functional-output dtype because its graph has no live parameter destination.
    Expert modules keep non-owning runtime references.

    Args:
        config: User-selected memory and pipeline-slot policy.
        model_parts: Final local model or pipeline-stage modules.
        parallelism_context: Final distributed mesh topology.
        device: CUDA device that owns the Annex context and buffers.
        num_tokens_per_microbatch_per_dp_rank: Unsharded token count used to
            derive the exact local routing-input shape after CP and TP.
        pp_schedule: Schedule used for activation-liveness analysis, or
            ``None`` without pipeline parallelism. Eager PP also executes
            metadata inference on this schedule.
        register_post_metadata_inference_cleanup: Optional registrar for the
            schedule that executes metadata inference when it differs from
            ``pp_schedule``. GraphPP uses this to retain its pre-rewrite
            liveness schedule while registering cleanup on its runtime schedule.
        set_forward_context: GraphPP-owned setter for its slot resolver. Eager
            PP leaves this unset and the runtime registers directly on each
            local stage. Passing ``None`` to the setter removes the GraphPP
            registration during cleanup.
        wgrad_dtype: Optional explicit functional WGrad dtype. Eager execution
            leaves this unset and lets the annex resolve live parameter-owned
            storage. GraphTrainer supplies the BF16 compute-parameter dtype
            because its functional graphs have no live parameter destination.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configure rank-wide Dist-MoE memory and execution policy.

        Args:
            activation_slot_bytes: Exact saved-forward-state capacity of each
                activation slot. It is mutually exclusive with
                ``activation_slot_capacity_factor``.
            activation_slot_capacity_factor: Optional saved-state capacity
                relative to balanced routing for each slot. ``1.0`` retains all
                eligible intermediates when aggregate routing across the layers
                assigned to a slot is balanced. Routing beyond the available
                capacity causes the annex to recompute affected intermediates.
                When both activation-capacity controls are ``None``, the annex
                allocates only mandatory inputs and recomputes intermediates.
            scratch_capacity_factor: Routing imbalance that must fit entirely
                in device-resident scratch. ``1.0`` covers balanced
                ``local_tokens * top_k`` routing.
            vmm_capacity_factor: Optional total device-plus-host scratch
                capacity. ``None`` disables VMM. A larger value lets routing
                above ``scratch_capacity_factor`` spill into host-backed pages.
            pp_activation_slot_policy: PP liveness granularity.
                ``"stage_microbatch"`` lets different local stages reuse slots
                as soon as each stage's backward releases its state.
                ``"microbatch"`` retains one slot across all local stages for a
                microbatch.
            inference: Specialize the context for a vLLM generator: scratch-only
                planning with no saved activations, and a per-step row count
                instead of the planned maximum. The activation-slot controls
                must stay ``None``; ``dist_moe`` rejects them for inference.
        """

        activation_slot_bytes: int | None = None
        activation_slot_capacity_factor: float | None = None
        scratch_capacity_factor: float = 1.0
        vmm_capacity_factor: float | None = None
        pp_activation_slot_policy: PPActivationSlotPolicy = "stage_microbatch"
        inference: bool = False

        def __post_init__(self) -> None:
            if (
                self.activation_slot_bytes is not None
                and self.activation_slot_capacity_factor is not None
            ):
                raise ValueError(
                    "activation_slot_bytes and activation_slot_capacity_factor "
                    "are mutually exclusive"
                )
            if self.activation_slot_bytes is not None:
                if isinstance(self.activation_slot_bytes, bool) or not isinstance(
                    self.activation_slot_bytes, int
                ):
                    raise TypeError("activation_slot_bytes must be an integer")
                if self.activation_slot_bytes < 0:
                    raise ValueError("activation_slot_bytes cannot be negative")
            if self.activation_slot_capacity_factor is not None:
                if isinstance(
                    self.activation_slot_capacity_factor, bool
                ) or not isinstance(self.activation_slot_capacity_factor, (int, float)):
                    raise TypeError(
                        "activation_slot_capacity_factor must be a real number"
                    )
                if (
                    not math.isfinite(self.activation_slot_capacity_factor)
                    or self.activation_slot_capacity_factor < 0
                ):
                    raise ValueError(
                        "activation_slot_capacity_factor must be finite and nonnegative"
                    )
            if (
                not math.isfinite(self.scratch_capacity_factor)
                or self.scratch_capacity_factor <= 0
            ):
                raise ValueError("scratch_capacity_factor must be finite and positive")
            if self.vmm_capacity_factor is not None and (
                not math.isfinite(self.vmm_capacity_factor)
                or self.vmm_capacity_factor <= 0
            ):
                raise ValueError("vmm_capacity_factor must be finite and positive")
            if self.pp_activation_slot_policy not in (
                "stage_microbatch",
                "microbatch",
            ):
                raise ValueError("unsupported PP activation-slot policy")

    def __init__(
        self,
        config: Config,
        *,
        model_parts: Sequence[torch.nn.Module],
        parallelism_context: ParallelismContext,
        device: torch.device,
        num_tokens_per_microbatch_per_dp_rank: int,
        pp_schedule: PipelineScheduleMulti | None,
        register_post_metadata_inference_cleanup: (
            Callable[[Callable[[], None]], RemovableHandle] | None
        ) = None,
        set_forward_context: (
            Callable[[_DistMoeForwardContext | None], None] | None
        ) = None,
        wgrad_dtype: torch.dtype | None = None,
    ) -> None:
        from .routed_experts import DistMoeRoutedExperts

        self.config = config
        self._num_local_input_tokens_per_call: int | None = None
        self._closed = False
        self._forward_context_handles: list[RemovableHandle] = []
        self._metadata_inference_cleanup_handle: RemovableHandle | None = None
        self._set_forward_context = set_forward_context
        self._modules = tuple(
            dict.fromkeys(
                module
                for model_part in model_parts
                for module in model_part.modules()
                if isinstance(module, DistMoeRoutedExperts)
            )
        )
        if not self._modules:
            raise ValueError("Dist-MoE runtime requires at least one expert module")
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
            raise ValueError("Dist-MoE requires an SM100-or-newer CUDA device")

        ep_mesh = parallelism_context.get_optional_mesh(
            "ep", include_singleton_axes=True
        )
        if ep_mesh is None:
            raise RuntimeError("Dist-MoE requires an expert-parallel mesh")
        ep_pg = ep_mesh.get_group()

        num_token_shards = parallelism_context.cp * parallelism_context.tp
        if num_tokens_per_microbatch_per_dp_rank % num_token_shards:
            raise ValueError(
                "Dist-MoE input tokens must divide evenly across CP and TP"
            )
        max_num_local_input_tokens = (
            num_tokens_per_microbatch_per_dp_rank // num_token_shards
        )

        max_live_activation_slots = 1
        max_moe_layers_per_activation_slot = len(self._modules)
        active_stage_indices = frozenset({0})
        activation_slot_id_by_stage_and_microbatch: dict[tuple[int, int], int] = {}
        if parallelism_context.pp_enabled:
            if pp_schedule is None:
                raise ValueError(
                    "Dist-MoE PP activation planning requires a multi-stage schedule"
                )
            pp_mesh = parallelism_context.get_optional_mesh(
                "pp", include_singleton_axes=True
            )
            if pp_mesh is None:
                raise RuntimeError("pipeline parallelism requires a PP mesh")
            plan = self._plan_pp_activation_slots(
                pp_schedule,
                pp_rank=pp_mesh.get_local_rank(),
                model_parts=model_parts,
            )
            max_live_activation_slots = plan.max_live_activation_slots
            max_moe_layers_per_activation_slot = plan.max_moe_layers_per_activation_slot
            active_stage_indices = frozenset(
                stage_index
                for stage_index, _ in plan.activation_slot_id_by_stage_and_microbatch
            )
            activation_slot_id_by_stage_and_microbatch = (
                plan.activation_slot_id_by_stage_and_microbatch
            )
            logger.info(
                "Dist-MoE PP activation slots: policy=%s slots=%d depth=%d",
                config.pp_activation_slot_policy,
                max_live_activation_slots,
                max_moe_layers_per_activation_slot,
            )

        activation_slot_ids_S = torch.arange(
            max_live_activation_slots,
            dtype=torch.int64,
            device=device,
        )

        context_config = self._resolve_context_config(
            self._modules[0],
            max_num_local_input_tokens=max_num_local_input_tokens,
            max_live_activation_slots=max_live_activation_slots,
            max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
            wgrad_dtype=wgrad_dtype,
        )
        for module in self._modules[1:]:
            candidate = self._resolve_context_config(
                module,
                max_num_local_input_tokens=max_num_local_input_tokens,
                max_live_activation_slots=max_live_activation_slots,
                max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
                wgrad_dtype=wgrad_dtype,
            )
            if candidate != context_config:
                raise ValueError(
                    "All local Dist-MoE layers must resolve one context configuration"
                )
        self.context = dist_moe.create_context(
            group=ep_pg,
            config=context_config,
            device=device,
        )
        self.forward_context = _DistMoeForwardContext(
            self.context,
            active_stage_indices=active_stage_indices,
            activation_slot_id_by_stage_and_microbatch=(
                activation_slot_id_by_stage_and_microbatch
            ),
            activation_slot_ids_S=activation_slot_ids_S,
            max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
        )
        try:
            if pp_schedule is not None:
                cleanup_registrar = register_post_metadata_inference_cleanup
                if cleanup_registrar is None:
                    cleanup_registrar = (
                        pp_schedule.register_post_metadata_inference_cleanup
                    )
                self._metadata_inference_cleanup_handle = cleanup_registrar(
                    self.context.reset
                )
                if set_forward_context is None:
                    for stage in pp_schedule._stages:
                        self._forward_context_handles.append(
                            stage.register_forward_context(self.forward_context)
                        )
                else:
                    set_forward_context(self.forward_context)
            for module in self._modules:
                module._runtime = self
        except Exception:
            if self._metadata_inference_cleanup_handle is not None:
                self._metadata_inference_cleanup_handle.remove()
                self._metadata_inference_cleanup_handle = None
            for handle in reversed(self._forward_context_handles):
                handle.remove()
            if set_forward_context is not None:
                set_forward_context(None)
            self.context.close()
            raise

    def _resolve_context_config(
        self,
        module: DistMoeRoutedExperts,
        *,
        max_num_local_input_tokens: int,
        max_live_activation_slots: int,
        max_moe_layers_per_activation_slot: int,
        wgrad_dtype: torch.dtype | None = None,
    ) -> dist_moe.Config:
        """Build the annex context configuration for one local expert module."""
        vmm = (
            None
            if self.config.vmm_capacity_factor is None
            else dist_moe.VmmConfig(
                total_scratch_capacity_factor=self.config.vmm_capacity_factor
            )
        )
        activation_options: dict[str, float] = {}
        if module.activation == "swiglu_clamped":
            assert module.swiglu_alpha is not None
            assert module.swiglu_limit is not None
            activation_options = {
                "swiglu_alpha": module.swiglu_alpha,
                "swiglu_limit": module.swiglu_limit,
            }
        return dist_moe.Config(
            max_num_local_input_tokens=max_num_local_input_tokens,
            hidden_dim=module.hidden_dim,
            intermediate_dim=module.intermediate_dim,
            top_k=module.top_k,
            num_experts=module.num_experts,
            max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
            device_scratch_capacity_factor=self.config.scratch_capacity_factor,
            activation_slot_bytes=self.config.activation_slot_bytes,
            activation_slot_capacity_factor=(
                self.config.activation_slot_capacity_factor
            ),
            num_activation_slots=max_live_activation_slots,
            vmm=vmm,
            bf16_grouped_gemm_preset=self._bf16_grouped_gemm_preset(module),
            block_scaled=module.block_scaled_config,
            activation=module.activation,
            wgrad_dtype=wgrad_dtype,
            inference=self.config.inference,
            **activation_options,
        )

    def _bf16_grouped_gemm_preset(self, module: DistMoeRoutedExperts) -> str | None:
        """Return the module's preset, or a fixed one for batch-invariant inference."""
        if (
            module.bf16_grouped_gemm_preset is None
            and module.block_scaled_config is None
            and self.config.inference
            and is_in_batch_invariant_mode()
        ):
            return _BATCH_INVARIANT_BF16_GROUPED_GEMM_PRESET
        return module.bf16_grouped_gemm_preset

    def _plan_pp_activation_slots(
        self,
        schedule: PipelineScheduleMulti,
        *,
        pp_rank: int,
        model_parts: Sequence[torch.nn.Module],
    ) -> _DistMoePipelineActivationPlan:
        """Derive immutable Dist-MoE slot assignments from the PP schedule."""
        from .routed_experts import DistMoeRoutedExperts

        if len(schedule._stages) != len(model_parts):
            raise RuntimeError("pipeline schedule and model parts disagree")
        modules_by_stage = {
            stage.stage_index: modules
            for stage, model_part in zip(schedule._stages, model_parts, strict=True)
            if (
                modules := [
                    module
                    for module in model_part.modules()
                    if isinstance(module, DistMoeRoutedExperts)
                ]
            )
        }
        if not modules_by_stage:
            raise RuntimeError("no local pipeline stage contains Dist-MoE")

        stage_indices = tuple(modules_by_stage)
        liveness = analyze_pipeline_activation_liveness(
            schedule,
            pp_rank=pp_rank,
            stage_indices=stage_indices,
            granularity=self.config.pp_activation_slot_policy,
        )
        if self.config.pp_activation_slot_policy == "microbatch":
            max_moe_layers_per_activation_slot = sum(
                len(modules) for modules in modules_by_stage.values()
            )
        else:
            max_moe_layers_per_activation_slot = max(
                len(modules) for modules in modules_by_stage.values()
            )
        activation_slot_id_by_stage_and_microbatch = {
            (stage_index, microbatch_index): liveness.slot_for(
                stage_index,
                microbatch_index,
            )
            for stage_index in stage_indices
            for microbatch_index in range(liveness.num_microbatches)
        }
        return _DistMoePipelineActivationPlan(
            max_live_activation_slots=liveness.num_slots,
            max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
            activation_slot_id_by_stage_and_microbatch=(
                activation_slot_id_by_stage_and_microbatch
            ),
        )

    def set_num_local_input_tokens_per_call(self, num_tokens: int) -> None:
        """Set the row count every EP rank passes to the annex in this step.

        Inference only. The caller must pass the same value on every EP rank,
        e.g. the largest TP-local token count across vLLM DP replicas, so a
        small decode step calls the annex at its own size.

        In batch-invariant mode small calls are raised above
        ``SCALE_AND_SUM_TILE_D_MAX_TOKENS`` rows: at or below it the annex's
        post-expert top-k reduction uses another, equally correct, accumulation
        order, so a token's output would depend on how many tokens share its
        step. The extra rows are ``-1`` routes, which cost no dispatch or expert
        work.
        """
        if not self.config.inference:
            raise RuntimeError("Only Dist-MoE inference supports a per-call size")
        max_num_tokens = self.context.max_num_local_input_tokens
        if not 1 <= num_tokens <= max_num_tokens:
            raise ValueError(
                f"Per-call Dist-MoE token count {num_tokens} must be in "
                f"[1, {max_num_tokens}]"
            )
        if is_in_batch_invariant_mode():
            from dist_moe.kernels.triton.broadcast_n_reduction import (
                SCALE_AND_SUM_TILE_D_MAX_TOKENS,
            )

            num_tokens = max(
                num_tokens, min(SCALE_AND_SUM_TILE_D_MAX_TOKENS + 1, max_num_tokens)
            )
        self._num_local_input_tokens_per_call = num_tokens

    def num_local_input_tokens_for_call(self, num_tokens: int) -> int:
        """Return the row count every EP rank passes to the annex for a call.

        Training passes its ``num_tokens`` local rows unchanged: every EP rank
        runs the same microbatch shape. Inference uses the count the generator
        set for this step, or the context maximum before the first step sets one.
        """
        if not self.config.inference:
            return num_tokens
        if self._num_local_input_tokens_per_call is None:
            return self.context.max_num_local_input_tokens
        return self._num_local_input_tokens_per_call

    def close(self) -> None:
        """Remove PP registrations, detach modules, and close Annex state."""
        if self._closed:
            return
        if self._metadata_inference_cleanup_handle is not None:
            self._metadata_inference_cleanup_handle.remove()
            self._metadata_inference_cleanup_handle = None
        for handle in reversed(self._forward_context_handles):
            handle.remove()
        self._forward_context_handles.clear()
        if self._set_forward_context is not None:
            self._set_forward_context(None)
        for module in self._modules:
            if module._runtime is self:
                module._runtime = None
        self.context.close()
        self._closed = True
