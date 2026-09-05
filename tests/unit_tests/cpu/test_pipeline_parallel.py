# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.pipeline_parallel import (
    _get_pipeline_metadata,
    _pipeline_module_split,
    pipeline_llm,
    PipelineRuntime,
    PipelineSharedParameterSpec,
    SharedParameterPipelineRuntime,
)
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.mtp import (
    _build_mtp_stage_metadata,
    _generate_mtp_fqn_per_model_part,
    _validate_mtp_fqn_per_model_part,
    MTPDecoder,
    MTPLoss,
    pipeline_deepseek_v3,
)


class _EmbeddingOwner(torch.nn.Module):
    """Minimal stage module with a shared token embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(4, 3)


class _TwoPartModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(2, 2)
        self.second = torch.nn.Linear(2, 2)


class _IdentityDecoderBlock(torch.nn.Module):
    def forward(self, hidden, attention_masks, positions):
        del attention_masks, positions
        return hidden


class _AddMTPBlock(torch.nn.Module):
    def forward(
        self,
        mtp_input_embed,
        prev_embed,
        mtp_input_valid_mask,
        attention_masks,
        positions,
    ):
        del attention_masks, positions
        return mtp_input_embed + prev_embed * mtp_input_valid_mask.unsqueeze(-1)


class _TestMTPDecoder(MTPDecoder):
    """Small MTP decoder assembled for pipeline contract testing."""

    def __init__(self, *, first: bool, final: bool) -> None:
        torch.nn.Module.__init__(self)
        self.num_mtp_layers = 1
        self._skip_lm_head = True
        self.tok_embeddings = torch.nn.Embedding(16, 4) if first or final else None
        self.layers = torch.nn.ModuleDict({"0": _IdentityDecoderBlock()})
        self.norm = torch.nn.Identity() if final else None
        self.lm_head = None
        self.mtp_layers = (
            torch.nn.ModuleList([_AddMTPBlock()]) if final else torch.nn.ModuleList()
        )


def test_default_pipeline_runtime_is_inert() -> None:
    runtime = PipelineRuntime()
    inputs = torch.ones(2, 3)
    kwargs = {"positions": torch.arange(3)}
    parameters = (torch.nn.Parameter(torch.ones(1)),)

    assert runtime.prepare_microbatch(inputs, kwargs) is kwargs
    assert runtime.parameters_for_grad_norm(parameters) == parameters
    runtime.synchronize_parameters()
    runtime.finalize_gradients()


def test_pipeline_split_uses_stage_metadata_fn() -> None:
    pp_mesh = MagicMock()
    pp_mesh.size.return_value = 1
    pp_mesh.get_local_rank.return_value = 0
    pp_mesh.get_group.return_value = MagicMock()
    first_input = torch.empty(2, dtype=torch.int64)
    first_output = torch.empty(2, 3)
    second_input = torch.empty(2, 3)
    second_output = torch.empty(2, 4)
    stage_metadata_fn = MagicMock(
        side_effect=((first_input, first_output), (second_input, second_output))
    )
    pipeline_stages = (MagicMock(), MagicMock())

    with patch(
        "torchtitan.distributed.pipeline_parallel.PipelineStage",
        side_effect=pipeline_stages,
    ) as pipeline_stage:
        stages, model_parts = _pipeline_module_split(
            _TwoPartModel(),
            pp_mesh,
            "Interleaved1F1B",
            torch.device("cpu"),
            [["first"], ["second"]],
            stage_metadata_fn=stage_metadata_fn,
        )

    assert stages == list(pipeline_stages)
    assert len(model_parts) == 2
    assert stage_metadata_fn.call_args_list[0].args == (0, 2)
    assert stage_metadata_fn.call_args_list[1].args == (1, 2)
    assert pipeline_stage.call_args_list[0].kwargs["input_args"] is first_input
    assert pipeline_stage.call_args_list[0].kwargs["output_args"] is first_output
    assert pipeline_stage.call_args_list[1].kwargs["input_args"] is second_input
    assert pipeline_stage.call_args_list[1].kwargs["output_args"] is second_output


def test_same_rank_shared_parameter_runtime_lifecycle() -> None:
    canonical = _EmbeddingOwner()
    replica = _EmbeddingOwner()
    canonical.tok_embeddings.weight.data.fill_(2.0)
    replica.tok_embeddings.weight.data.zero_()
    pp_mesh = MagicMock()
    pp_mesh.size.return_value = 1
    pp_mesh.get_local_rank.return_value = 0
    pp_mesh.get_group.return_value = MagicMock()
    runtime = SharedParameterPipelineRuntime(
        model_parts=[canonical, replica],
        stage_indices=(0, 2),
        pp_mesh=pp_mesh,
        pp_schedule="Interleaved1F1B",
        num_stages=4,
        shared_parameter_specs=(
            PipelineSharedParameterSpec(
                fqn="tok_embeddings.weight",
                stage_indices=(0, 2),
            ),
        ),
    )

    runtime.synchronize_parameters()
    torch.testing.assert_close(
        replica.tok_embeddings.weight,
        canonical.tok_embeddings.weight,
    )

    canonical.tok_embeddings.weight.grad = torch.full_like(
        canonical.tok_embeddings.weight, 3.0
    )
    replica.tok_embeddings.weight.grad = torch.full_like(
        replica.tok_embeddings.weight, 5.0
    )
    runtime.finalize_gradients()
    expected_grad = torch.full_like(canonical.tok_embeddings.weight, 8.0)
    torch.testing.assert_close(canonical.tok_embeddings.weight.grad, expected_grad)
    torch.testing.assert_close(replica.tok_embeddings.weight.grad, expected_grad)

    parameters = tuple(canonical.parameters()) + tuple(replica.parameters())
    assert runtime.parameters_for_grad_norm(parameters) == tuple(canonical.parameters())


def test_shared_parameter_runtime_initializes_parent_before_split() -> None:
    pp_mesh = MagicMock()
    pp_mesh.size.return_value = 4
    pp_mesh.get_local_rank.return_value = 1
    parent_group = MagicMock()
    pp_mesh.get_group.return_value = parent_group
    calls = []

    with (
        patch.object(
            dist,
            "barrier",
            side_effect=lambda **kwargs: calls.append(("barrier", kwargs)),
        ),
        patch.object(
            dist,
            "split_group",
            side_effect=lambda **kwargs: (
                calls.append(("split_group", kwargs))
                or dist.GroupMember.NON_GROUP_MEMBER
            ),
        ),
        patch(
            "torchtitan.distributed.pipeline_parallel.device_module.current_device",
            return_value=2,
        ),
    ):
        SharedParameterPipelineRuntime(
            model_parts=[_EmbeddingOwner()],
            stage_indices=(1,),
            pp_mesh=pp_mesh,
            pp_schedule="Interleaved1F1B",
            num_stages=4,
            shared_parameter_specs=(
                PipelineSharedParameterSpec(
                    fqn="tok_embeddings.weight",
                    stage_indices=(0, 3),
                ),
            ),
        )

    assert [name for name, _ in calls] == ["barrier", "split_group"]
    assert calls[0][1] == {"group": parent_group, "device_ids": [2]}
    assert calls[1][1]["parent_pg"] is parent_group
    assert calls[1][1]["split_ranks"] == [[0, 3]]


def test_two_rank_runtime_broadcasts_from_canonical_owner() -> None:
    canonical = _EmbeddingOwner()
    pp_mesh = MagicMock()
    pp_mesh.size.return_value = 2
    pp_mesh.get_local_rank.return_value = 1
    pp_group = MagicMock()
    pp_mesh.get_group.return_value = pp_group
    runtime = SharedParameterPipelineRuntime(
        model_parts=[canonical],
        stage_indices=(1,),
        pp_mesh=pp_mesh,
        pp_schedule="1F1B",
        num_stages=2,
        shared_parameter_specs=(
            PipelineSharedParameterSpec(
                fqn="tok_embeddings.weight",
                stage_indices=(1, 0),
            ),
        ),
    )

    with patch.object(dist, "broadcast") as broadcast:
        runtime.synchronize_parameters()

    broadcast.assert_called_once_with(
        canonical.tok_embeddings.weight,
        group=pp_group,
        group_src=1,
    )


def test_cross_rank_runtime_reduces_shared_gradient() -> None:
    replica = _EmbeddingOwner()
    replica.tok_embeddings.weight.grad = torch.ones_like(replica.tok_embeddings.weight)
    pp_mesh = MagicMock()
    pp_mesh.size.return_value = 2
    pp_mesh.get_local_rank.return_value = 0
    pp_group = MagicMock()
    pp_mesh.get_group.return_value = pp_group
    runtime = SharedParameterPipelineRuntime(
        model_parts=[replica],
        stage_indices=(0,),
        pp_mesh=pp_mesh,
        pp_schedule="1F1B",
        num_stages=2,
        shared_parameter_specs=(
            PipelineSharedParameterSpec(
                fqn="tok_embeddings.weight",
                stage_indices=(1, 0),
            ),
        ),
    )

    with patch.object(dist, "all_reduce") as all_reduce:
        runtime.finalize_gradients()

    all_reduce.assert_called_once_with(
        replica.tok_embeddings.weight.grad,
        group=pp_group,
    )


def test_generate_mtp_fqn_per_model_part_places_final_stage_ownership() -> None:
    model_config = model_registry("debugmodel", num_mtp_layers=1).model
    layout = _generate_mtp_fqn_per_model_part(
        model_config,
        num_stages=4,
        num_layers=6,
        input_weight=1,
        output_weight=1,
    )

    assert layout[0].count("tok_embeddings") == 1
    assert layout[-1][-2:] == ["mtp_layers.0", "tok_embeddings"]
    _validate_mtp_fqn_per_model_part(model_config, layout)

    invalid = [list(stage) for stage in layout]
    invalid[0].append(invalid[-1].pop(-2))
    with pytest.raises(ValueError, match="must belong only to final stage"):
        _validate_mtp_fqn_per_model_part(model_config, invalid)


def test_mtp_model_registry_selects_mtp_pipeline_builder() -> None:
    model_spec = model_registry("debugmodel", num_mtp_layers=1)

    assert model_spec.pipelining_fn is pipeline_deepseek_v3
    assert model_registry("debugmodel").pipelining_fn is pipeline_llm


def test_explicit_pipeline_layout_defines_virtual_stage_count() -> None:
    model_config = model_registry("debugmodel").model
    parallelism = ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_layers_per_stage=None,
        module_fqns_per_model_part=[["layers.0"] for _ in range(8)],
    )

    num_stages, *_ = _get_pipeline_metadata(
        MagicMock(pp=4),
        parallelism,
        model_config,
    )

    assert num_stages == 8


def test_mtp_stage_metadata_matches_pipeline_contract() -> None:
    model_config = model_registry("debugmodel", num_mtp_layers=1).model
    training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=8,
        mixed_precision_param="float32",
    )
    chunked_loss = ChunkedLossWrapper(
        ChunkedLossWrapper.Config(
            loss_fn=MTPLoss.Config(global_vocab_size=model_config.vocab_size),
        )
    )

    first_input, first_output = _build_mtp_stage_metadata(
        0,
        4,
        training=training,
        model_config=model_config,
        loss_fn=chunked_loss,
    )
    assert first_input.shape == (8,)
    assert first_input.dtype == torch.int64
    assert first_output.shape == (8, model_config.dim)

    last_input, last_output = _build_mtp_stage_metadata(
        3,
        4,
        training=training,
        model_config=model_config,
        loss_fn=chunked_loss,
    )
    assert last_input.shape == (8, model_config.dim)
    assert isinstance(last_output, tuple)
    assert len(last_output) == 3
    assert last_output[0].shape == (8, model_config.dim)
    assert last_output[1].shape == (8, model_config.dim)
    assert last_output[2].shape == (8,)
    assert last_output[2].dtype == torch.bool


def test_mtp_staged_forward_keeps_tokens_off_pipeline_edges() -> None:
    torch.manual_seed(42)
    full = _TestMTPDecoder(first=True, final=True)
    first = _TestMTPDecoder(first=True, final=False)
    final = _TestMTPDecoder(first=False, final=True)
    first.tok_embeddings.load_state_dict(full.tok_embeddings.state_dict())
    final.tok_embeddings.load_state_dict(full.tok_embeddings.state_dict())
    tokens = torch.tensor([1, 2, 3, 4])
    positions = torch.arange(4)

    expected = full(tokens, positions=positions)
    hidden = first(tokens, positions=positions, mtp_source_tokens=tokens)
    assert isinstance(hidden, torch.Tensor)
    actual = final(
        hidden,
        positions=positions,
        mtp_source_tokens=tokens,
    )

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
