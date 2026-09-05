# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from torchtitan.distributed.pipeline_parallel import (
    _pipeline_module_split,
    PipelineRuntime,
    PipelineSharedParameterSpec,
    SharedParameterPipelineRuntime,
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


def test_default_pipeline_runtime_is_inert() -> None:
    runtime = PipelineRuntime()
    inputs = torch.ones(2, 3)
    kwargs = {"positions": torch.arange(3)}
    parameters = (torch.nn.Parameter(torch.ones(1)),)

    assert runtime.prepare_microbatch(inputs, kwargs) is kwargs
    assert runtime.parameters_for_grad_norm(parameters) == parameters
    runtime.synchronize_parameters()
    runtime.finalize_gradients()


def test_pipeline_split_uses_model_stage_args() -> None:
    pp_mesh = MagicMock()
    pp_mesh.size.return_value = 1
    pp_mesh.get_local_rank.return_value = 0
    pp_group = MagicMock()
    pp_mesh.get_group.return_value = pp_group
    first_input = torch.empty(2, dtype=torch.int64)
    first_output = torch.empty(2, 3)
    second_input = torch.empty(2, 3)
    second_output = torch.empty(2, 4)
    stage_args = MagicMock(
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
            static_stage_args=stage_args,
        )

    assert stages == list(pipeline_stages)
    assert len(model_parts) == 2
    assert stage_args.call_args_list[0].args == (0, 2)
    assert stage_args.call_args_list[1].args == (1, 2)
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
