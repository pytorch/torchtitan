# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    ScheduleZBVZeroBubble,
)
from transformers import PretrainedConfig

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline import (
    build_pipeline_schedule,
    generate_split_points,
    stage_ids_this_rank,
)
from torchtitan.tools.logging import logger

DeviceType = Union[int, str, torch.device]


def patch_llama_forward(model: nn.Module):
    """
    patch forward method for LlamaModel instance.
    Revised from transformers.models.llama.modeling_llama.LlamaModel.forward
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> torch.Tensor:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, False
        )

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers, ok since ModuleDict is ordered
        for decoder_layer in list(self.layers.values())[
            : self.config.num_hidden_layers
        ]:

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = layer_outputs[0]

        # adapt for last stage / other stage
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    # bind to model instance
    model.forward = forward.__get__(model, model.__class__)


def patch_llamaforcasuallm_forward(model: nn.Module):
    """
    patch forward method for LlamaForCausalLM instance
    """

    def forward(
        self,
        input_ids_or_input_embeds: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> torch.Tensor:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if return_dict is True:
            raise ValueError(
                "return_dict must be False while using pipeline parallelism"
            )

        # adapt for first staget / other stage
        if self.model.embed_tokens is not None:
            input_ids = input_ids_or_input_embeds
            inputs_embeds = None
        else:
            input_ids = None
            inputs_embeds = input_ids_or_input_embeds

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        # adapt for last stage / other stage
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return (logits,)
        else:
            return (hidden_states[:, slice_indices, :],)

    # bind to model instance
    model.forward = forward.__get__(model, model.__class__)

    # patch for LlamaModel instance
    patch_llama_forward(model.model)


def pipeline_llama(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: PretrainedConfig,
    loss_fn: Callable[..., torch.Tensor],
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    logger.info("Changing model.model.layers to nn.ModuleDict")
    model.model.layers = nn.ModuleDict(
        {str(i): layer for i, layer in enumerate(model.model.layers)}
    )
    logger.info(
        "Patching Llama forward method for pipeline parallelism, it will disable some features of orignal HF model"
    )
    patch_llamaforcasuallm_forward(model)
    stages, models = pipeline_llama_manual_split(
        model, pp_mesh, parallel_dims, job_config, device, model_config
    )

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, models, has_first_stage, has_last_stage


def pipeline_llama_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: PretrainedConfig,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    splits = (
        job_config.experimental.pipeline_parallel_split_points
        or generate_split_points(
            job_config, parallel_dims.pp, model_config.num_hidden_layers
        )
    )

    def _build_stage(
        stage_idx: int,
        start_layer: Optional[str],
        stop_layer: Optional[str],
        is_first: bool = False,
        is_last: bool = False,
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.model.embed_tokens = None

        drop_layers = start_layer is not None
        for name in list(model.model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.model.layers[name]

        if not is_last:
            model.model.norm = None
            model.lm_head = None

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []

    schedule_class = get_schedule_class(
        job_config.experimental.pipeline_parallel_schedule
    )
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style=style):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models
