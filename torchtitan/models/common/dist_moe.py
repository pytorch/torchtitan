# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed routed experts backed by the standalone ``dist_moe`` package.

Shape suffixes in this file use ``T`` for local input tokens, ``K`` for selected
experts, ``E`` for local experts, ``F`` for the expert intermediate dimension,
and ``D`` for the model dimension.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, Literal, TYPE_CHECKING

import dist_moe
import torch
import torch_remat as remat
from torch.distributed.pipelining import (
    analyze_pipeline_activation_liveness,
    PipelineStageInfo,
)
from torch.distributed.pipelining.schedules import PipelineScheduleMulti

from torchtitan.components.runtime import TrainingRuntime
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module


if TYPE_CHECKING:
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.training_engine import TrainingEngine


logger = logging.getLogger(__name__)

__all__ = ["DistMoeRoutedExperts", "DistMoeRuntime"]

PPActivationSlotPolicy = Literal["stage_microbatch", "microbatch"]
_DistMoeWeightOperand = torch.Tensor | dist_moe.PreparedWeight


@dataclass(frozen=True, slots=True)
class _DistMoePipelineActivationPlan:
    """Static activation-slot assignment for Dist-MoE stages on one PP rank.

    PyTorch's schedule analysis colors the live interval from each stage forward
    through the backward action that releases its saved state. The resulting
    slot IDs are reused only for non-overlapping intervals. Dist-MoE additionally
    records the maximum MoE-layer depth assigned to a slot because the annex
    planner stores all those layers in the same slot.

    Attributes:
        max_live_activation_slots: Number of physical slots required by the
            schedule.
        max_moe_layers_per_activation_slot: Maximum local MoE-layer depth that
            can append saved state to one slot.
        activation_slot_by_stage_and_microbatch: Mapping from a global stage and
            microbatch pair to ``(slot_id, stage_moe_depth)``.
    """

    max_live_activation_slots: int
    max_moe_layers_per_activation_slot: int
    activation_slot_by_stage_and_microbatch: dict[tuple[int, int], tuple[int, int]]


class DistMoeRuntime(TrainingRuntime):
    """Own one annex context shared by all local Dist-MoE expert modules.

    The runtime is prepared after model parallelization because its memory plan
    depends on the final local stages, expert-parallel process group, and PP
    schedule. It initializes the annex context only after model parameters and
    buffers materialize. Expert modules keep non-owning references to this
    runtime and use its context during forward.
    """

    config: "DistMoeRuntime.Config"
    context: dist_moe.Context | None
    context_device: torch.device
    ep_pg: torch.distributed.ProcessGroup

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingRuntime.Config):
        """Configure rank-wide Dist-MoE memory and execution policy.

        Args:
            device_scratch_capacity_factor: Per-layer routing imbalance that
                must fit in device-resident scratch. ``1.0`` covers balanced
                ``local_tokens * top_k`` routing. When VMM is enabled, larger
                imbalances may use host-backed overflow up to
                ``vmm.total_scratch_capacity_factor``.
            activation_slot_bytes: Exact saved-forward-state capacity of each
                activation slot. ``None`` selects the annex's minimum
                all-recompute budget unless
                ``activation_slot_capacity_factor`` is set. Under PP the
                schedule determines how many slots are live; without PP one
                slot is used.
            activation_slot_capacity_factor: Saved-state capacity relative to
                balanced routing for each slot. ``1.0`` can retain every
                eligible intermediate when aggregate slot usage is balanced.
                Exceeding this soft capacity recomputes the affected layer.
                Mutually exclusive with ``activation_slot_bytes``.
            pp_activation_slot_policy: PP liveness granularity.
                ``"stage_microbatch"`` lets different local stages reuse slots
                as soon as each stage's backward releases its state.
                ``"microbatch"`` retains one slot across all local stages for a
                microbatch.
            vmm: Optional annex policy for host-backed overflow scratch. VMM
                never stores saved activations in host memory.
            num_sms: Optional SM count used by each Dist-MoE CuTe launch. Leave
                unset to use the annex default.
            wgrad_dtype: Dtype produced for W13 and W2 gradients.
        """

        device_scratch_capacity_factor: float = 1.0
        activation_slot_bytes: int | None = None
        activation_slot_capacity_factor: float | None = None
        pp_activation_slot_policy: PPActivationSlotPolicy = "stage_microbatch"
        vmm: dist_moe.VmmConfig | None = None
        num_sms: int | None = None
        wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"

        def __post_init__(self) -> None:
            if self.device_scratch_capacity_factor <= 0:
                raise ValueError("device_scratch_capacity_factor must be positive")
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
                self.activation_slot_bytes is not None
                and self.activation_slot_capacity_factor is not None
            ):
                raise ValueError(
                    "activation_slot_bytes and activation_slot_capacity_factor "
                    "are mutually exclusive"
                )
            if self.pp_activation_slot_policy not in (
                "stage_microbatch",
                "microbatch",
            ):
                raise ValueError("unsupported PP activation-slot policy")
            if self.num_sms is not None and self.num_sms <= 0:
                raise ValueError("num_sms must be positive")
            if self.wgrad_dtype not in ("bfloat16", "float32"):
                raise ValueError("unsupported Dist-MoE WGRAD dtype")

        def validate(self, training_config: object) -> None:
            """Validate mixed precision required by Dist-MoE parameters."""
            training = getattr(training_config, "training", None)
            if training is None or training.mixed_precision_param != "bfloat16":
                raise ValueError("Dist-MoE requires mixed_precision_param='bfloat16'")

    def __init__(
        self,
        config: Config,
        *,
        trainer_config: TrainingEngine.Config,
        model_parts: Sequence[torch.nn.Module],
        parallelism_context: ParallelismContext,
        device: torch.device,
        pp_schedule: object | None,
    ) -> None:
        self.config = config
        self.context: dist_moe.Context | None = None
        # pyrefly: ignore [read-only]
        self.context_device = device
        self._modules = tuple(
            dict.fromkeys(
                module
                for model_part in model_parts
                for module in model_part.modules()
                if isinstance(module, DistMoeRoutedExperts)
            )
        )
        self.pp_activation_slot_by_stage_and_microbatch: dict[
            tuple[int, int], tuple[int, int]
        ] = {}
        self._context_config: dist_moe.Config | None = None

        if not self._modules:
            return
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
            raise ValueError("Dist-MoE requires an SM100-or-newer CUDA device")

        ep_mesh = parallelism_context.get_optional_mesh(
            "ep", include_singleton_axes=True
        )
        if ep_mesh is None:
            raise RuntimeError("Dist-MoE requires an expert-parallel mesh")
        self.ep_pg = ep_mesh.get_group()

        num_local_tokens = trainer_config.training.num_tokens_per_microbatch_per_dp_rank
        num_token_shards = parallelism_context.cp * parallelism_context.tp
        if num_local_tokens % num_token_shards:
            raise ValueError(
                "Dist-MoE input tokens must divide evenly across CP and TP"
            )
        max_local_input_tokens = num_local_tokens // num_token_shards

        max_live_activation_slots = 1
        max_moe_layers_per_activation_slot = len(self._modules)
        if parallelism_context.pp_enabled:
            if not isinstance(pp_schedule, PipelineScheduleMulti):
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
            self.pp_activation_slot_by_stage_and_microbatch = (
                plan.activation_slot_by_stage_and_microbatch
            )
            logger.info(
                "Dist-MoE PP activation slots: policy=%s slots=%d depth=%d",
                config.pp_activation_slot_policy,
                max_live_activation_slots,
                max_moe_layers_per_activation_slot,
            )

        self._context_config = self._resolve_context_config(
            self._modules[0],
            max_local_input_tokens=max_local_input_tokens,
            max_live_activation_slots=max_live_activation_slots,
            max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
        )
        for module in self._modules[1:]:
            candidate = self._resolve_context_config(
                module,
                max_local_input_tokens=max_local_input_tokens,
                max_live_activation_slots=max_live_activation_slots,
                max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
            )
            if candidate != self._context_config:
                raise ValueError(
                    "All local Dist-MoE layers must resolve one context configuration"
                )
        for module in self._modules:
            module._runtime = self

    def _resolve_context_config(
        self,
        module: DistMoeRoutedExperts,
        *,
        max_local_input_tokens: int,
        max_live_activation_slots: int,
        max_moe_layers_per_activation_slot: int,
    ) -> dist_moe.Config:
        """Build the annex context configuration for one local expert module."""
        return dist_moe.Config(
            max_local_input_tokens=max_local_input_tokens,
            hidden_dim=module.hidden_dim,
            intermediate_dim=module.intermediate_dim,
            top_k=module.top_k,
            num_experts=module.num_experts,
            max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
            device_scratch_capacity_factor=self.config.device_scratch_capacity_factor,
            activation_slot_bytes=self.config.activation_slot_bytes,
            activation_slot_capacity_factor=(
                self.config.activation_slot_capacity_factor
            ),
            num_activation_slots=max_live_activation_slots,
            vmm=self.config.vmm,
            num_sms=self.config.num_sms,
            bf16_grouped_gemm_preset=module.bf16_grouped_gemm_preset,
            block_scaled=module.block_scaled_config,
            wgrad_dtype=(
                torch.bfloat16
                if self.config.wgrad_dtype == "bfloat16"
                else torch.float32
            ),
        )

    def _plan_pp_activation_slots(
        self,
        schedule: PipelineScheduleMulti,
        *,
        pp_rank: int,
        model_parts: Sequence[torch.nn.Module],
    ) -> _DistMoePipelineActivationPlan:
        """Derive immutable Dist-MoE slot assignments from the PP schedule."""
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
        activation_slot_by_stage_and_microbatch = {
            (stage_index, microbatch_index): (
                liveness.slot_for(stage_index, microbatch_index),
                (
                    max_moe_layers_per_activation_slot
                    if self.config.pp_activation_slot_policy == "microbatch"
                    else len(modules_by_stage[stage_index])
                ),
            )
            for stage_index in stage_indices
            for microbatch_index in range(liveness.num_microbatches)
        }
        return _DistMoePipelineActivationPlan(
            max_live_activation_slots=liveness.num_slots,
            max_moe_layers_per_activation_slot=(max_moe_layers_per_activation_slot),
            activation_slot_by_stage_and_microbatch=(
                activation_slot_by_stage_and_microbatch
            ),
        )

    def initialize(self) -> None:
        """Create the annex context after model state has materialized."""
        if self.context is not None or self._context_config is None:
            return
        self.context = dist_moe.create_context(
            group=self.ep_pg,
            config=self._context_config,
            device=self.context_device,
        )

    @contextmanager
    def forward_context(self, info: PipelineStageInfo) -> Iterator[None]:
        """Select the configured slot for one Dist-MoE PP forward action."""
        key = (info.stage_index, info.microbatch_index)
        selection = self.pp_activation_slot_by_stage_and_microbatch.get(key)
        if selection is not None:
            context = self.context
            if context is None:
                raise RuntimeError("Dist-MoE context is not initialized")
            activation_slot_id, num_moe_layers_in_slot = selection
            context.select_activation_slot(
                activation_slot_id,
                num_moe_layers_in_slot,
            )
        yield

    def forward_context_key(self, info: PipelineStageInfo) -> object | None:
        """Return the immutable activation-slot state bound during tracing."""
        return self.pp_activation_slot_by_stage_and_microbatch.get(
            (info.stage_index, info.microbatch_index)
        )

    def close(self) -> None:
        """Release the annex context and detach all module references."""
        context = self.context
        if context is not None:
            context.close()
            self.context = None
        for module in self._modules:
            module._runtime = None


class DistMoeRoutedExperts(RoutedExperts):
    """BF16 routed experts executed by the standalone Dist-MoE backend.

    The inherited W13 and W2 configs preserve ordinary TorchTitan parameter,
    FSDP, optimizer, and checkpoint ownership. Their module ``forward`` methods
    are not called: Dist-MoE consumes the weights directly and owns dispatch,
    SwiGLU, expert GEMMs, and combine. ``output_postprocess`` remains a normal
    TorchTitan module and is translated to an annex execution descriptor at the
    current FSDP unshard lifetime.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        """Configure BF16-specific Dist-MoE execution.

        Args:
            inplace_wgrad_accum: Whether Dist-MoE writes each W13/W2 gradient
                directly into an existing standard ``parameter.grad`` buffer.
                Dist-MoE derives the gradient owner from each logical weight;
                TorchTitan does not pass a separate owner. GraphTrainer keeps
                this disabled until its graph-owned accumulation pass can
                select the annex's mutating backward operations.
            bf16_grouped_gemm_preset: Optional expert override for the annex's
                BF16 FPROP/DGRAD grouped-GEMM schedule. ``None`` selects the
                shape-aware production defaults; WGRAD uses its independent
                production schedule.

        The inherited ``w13`` and ``w2`` configs define parameter shapes and
        checkpoint keys. The inherited dispatcher and SwiGLU configs describe
        the stock source module accepted by the transform but are not executed
        after replacement.
        """

        uses_configured_token_dispatcher: ClassVar[bool] = False

        inplace_wgrad_accum: bool = False
        bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None

        def __post_init__(self) -> None:
            """Validate the source-module contract required by Dist-MoE."""
            RoutedExperts.Config.__post_init__(self)
            if (
                type(self.w13) is not GroupedLinear.Config
                or type(self.w2) is not GroupedLinear.Config
                or type(self.activation_fn) is not SwiGLU.Config
            ):
                raise TypeError(
                    "Dist-MoE requires stock GroupedLinear W13/W2 projections "
                    "and SwiGLU"
                )
            if not isinstance(self.token_dispatcher, AllToAllTokenDispatcher.Config):
                raise ValueError(
                    "Dist-MoE requires the standard all-to-all source config; "
                    "the annex replaces its runtime dispatch and combine"
                )
            postprocess_config = self.output_postprocess
            owner = None if postprocess_config is None else postprocess_config._owner
            if postprocess_config is not None and not callable(
                getattr(owner, "to_dist_moe_postprocess", None)
            ):
                raise TypeError(
                    f"{type(postprocess_config).__qualname__} cannot execute "
                    "inside Dist-MoE"
                )

    def __init__(self, config: Config):
        Module.__init__(self)
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.output_postprocess = (
            config.output_postprocess.build()
            if config.output_postprocess is not None
            else None
        )
        self.hidden_dim = config.w13.in_features
        self.intermediate_dim = config.w2.in_features
        self.num_experts = config.w13.group_size
        self.top_k = config.token_dispatcher.top_k
        self.inplace_wgrad_accum = config.inplace_wgrad_accum
        self.bf16_grouped_gemm_preset = config.bf16_grouped_gemm_preset
        self.block_scaled_config: dist_moe.BlockScaledConfig | None = None
        self._runtime: DistMoeRuntime | None = None

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Leave communication and activation storage to the shared runtime."""
        del buffer_device

    def _weight_operands(
        self,
    ) -> tuple[_DistMoeWeightOperand, _DistMoeWeightOperand]:
        """Return W13 and W2 operands for the annex invocation."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        return w13_EFD, w2_EDF

    def _output_postprocess(self) -> dist_moe.RMSNormPostprocess | None:
        """Bind the current TorchTitan postprocess parameters to the annex."""
        module = self.output_postprocess
        if module is None:
            return None
        factory = getattr(module, "to_dist_moe_postprocess", None)
        if not callable(factory):
            raise TypeError(
                f"{type(module).__qualname__} cannot execute inside Dist-MoE"
            )
        postprocess = factory()
        if not isinstance(postprocess, dist_moe.RMSNormPostprocess):
            raise TypeError(
                "to_dist_moe_postprocess() must return dist_moe.RMSNormPostprocess"
            )
        return postprocess

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run distributed dispatch, expert computation, and combine.

        Args:
            x_TD: Local input tokens with model dimension ``D``.
            topk_scores_TK: Selected routing weights for ``K`` experts.
            topk_expert_ids_TK: Selected global expert IDs.
            num_local_tokens_per_expert_E: Router statistics retained by the
                surrounding MoE module; Dist-MoE derives dispatch metadata from
                the selected IDs.

        Returns:
            Combined local expert output with shape ``(T, D)``.
        """
        del num_local_tokens_per_expert_E
        runtime = self._runtime
        if runtime is None or runtime.context is None:
            raise RuntimeError("Dist-MoE context is not initialized")
        w13_operand, w2_operand = self._weight_operands()
        # TODO(graph_trainer): Add a WGRAD fusion rule that replaces functional
        # Dist-MoE backward outputs and their accumulation sinks with the
        # annex's graph-visible accumulating backward operations.
        execution_options = dist_moe.ExecutionOptions(
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            experts_output_postprocess=self._output_postprocess(),
        )
        out_TD = remat.region(
            dist_moe.routed_experts,
            self.remat_region_name("dist_moe"),
            recompute=False,
        )(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13_operand,
            w2_operand,
            runtime.context,
            options=execution_options,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD
