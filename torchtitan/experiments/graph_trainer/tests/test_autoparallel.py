# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._functorch.aot_autograd import (
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)

from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
        self.mesh_dim_names = tuple(mesh_axis_names)
        self.ndim = len(self.mesh_dim_names)
        self._size = size

    def size(self):
        return self._size


class _FakeParallelismContext:
    dp_replicate_enabled = False
    cp_enabled = False
    pp_enabled = False
    tp_enabled = True
    dp_replicate = 1
    dp_shard = 2

    def __init__(self, *, sparse: bool = False):
        self.sparse = sparse
        self.tp_enabled = not sparse

    def get_optional_mesh(self, name):
        enabled = {"dp_shard", "tp"} if not self.sparse else {"edp_shard", "ep"}
        return _FakeMesh((name,)) if name in enabled else None

    def get_mesh(self, names):
        if isinstance(names, str):
            return _FakeMesh((names,))
        return _FakeMesh(tuple(names))


class _FakeAutoParallelGraph:
    instances = []

    def __init__(self, model, input_fn, mesh, **kwargs):
        self.model = model
        self.input_fn = input_fn
        self.kwargs = kwargs
        self.used_fx_path = False
        _FakeAutoParallelGraph.instances.append(self)

    def __enter__(self):
        self.input_fn()
        self.gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_parameter_memory_constraint(self, *, low, high):
        pass

    def add_input_constraints(self, constraints):
        pass

    def add_output_constraints(self, constraints):
        pass

    def optimize_placement(self, verbose=False):
        return object()

    def apply_placement_for_fx_module(self, *args, **kwargs):
        self.used_fx_path = True
        self.apply_kwargs = kwargs
        return torch.nn.Linear(1, 1)


def _training_config():
    return TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=2 * 8,
    )


def test_autoparallel_graph_pass_selection_uses_regular_memory_policy():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            enable_async_tensor_parallel=False,
        ),
        model=SimpleNamespace(layers=[SimpleNamespace(feed_forward=None)]),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
        training=TrainingConfig(disable_accelerator_graphs=True),
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass in pass_fns


@pytest.mark.parametrize(
    (
        "model_name",
        "inductor_compilation",
        "fsdp_reshard_after_forward",
        "expected_reshard_after_forward",
    ),
    [
        ("llama", "regional", "always", True),
        ("deepseek", "full", "never", False),
    ],
)
def test_model_autoparallel_uses_fx_module_path_and_resolved_policy(
    model_name,
    inductor_compilation,
    fsdp_reshard_after_forward,
    expected_reshard_after_forward,
):
    class FakeDeepSeekV3Model(torch.nn.Module):
        def __init__(self, config, *, mesh, compute_dtype):
            super().__init__()
            assert config.layers[0].feed_forward is not None
            assert config.layers[0].moe is None
            assert config.layers[1].feed_forward is None
            assert config.layers[1].moe is not None
            self.model_args = SimpleNamespace(vocab_size=16)

    _FakeAutoParallelGraph.instances.clear()
    compile_config = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        inductor_compilation=inductor_compilation,
    )
    parallelism = ParallelismConfig(
        fsdp_reshard_after_forward=fsdp_reshard_after_forward
    )

    if model_name == "llama":
        from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

        model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=16, max_context_length=8),
            _get_attention_metadata=lambda *args, **kwargs: {},
        )
        parallelism_context = _FakeParallelismContext()
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_llama
        extra_patches = ()
    else:
        from torchtitan.experiments.graph_trainer.deepseek_v3 import (
            parallelize_autoparallel,
        )

        from torchtitan.models.deepseek_v3 import build_model_config

        model = SimpleNamespace(
            config=build_model_config("debugmodel", parallelism=ParallelismConfig())
        )
        parallelism_context = _FakeParallelismContext(sparse=True)
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_deepseekv3
        extra_patches = (
            patch.object(
                parallelize_autoparallel,
                "_load_autoparallel_dsv3_dependency",
                return_value=(FakeDeepSeekV3Model, lambda model: None),
            ),
        )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
            )
        )
        stack.enter_context(
            patch.object(
                parallelize_autoparallel, "apply_compile", lambda model, **_: model
            )
        )
        stack.enter_context(
            patch.object(parallelize_autoparallel, "device_type", "cpu")
        )
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)

        call_parallelize(
            model,
            parallelism_context=parallelism_context,
            precision=PrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
            training=_training_config(),
            parallelism=parallelism,
            compile_config=compile_config,
            activation_remat=object(),
            dump_folder="",
            max_num_documents=4,
        )

    autop = _FakeAutoParallelGraph.instances[0]
    mp_policy = autop.kwargs["mp_policy"]
    assert mp_policy.param_dtype is torch.bfloat16
    assert mp_policy.reduce_dtype is torch.float32
    assert autop.kwargs["reshard_after_forward"] is expected_reshard_after_forward
    assert autop.kwargs.get("dynamic", False) is (model_name == "deepseek")
    assert autop.apply_kwargs["compile_config"] is compile_config
    assert autop.used_fx_path


class _MetadataModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 3)

    def forward(self, tokens, *, positions, attention_metadata):
        offsets = attention_metadata[VarlenInnerAttention].cu_seq_q
        return self.projection(tokens) + (positions + offsets[1])[:, None]


def _metadata_kwargs(first_document_length):
    offsets = torch.tensor([0, first_document_length, 4], dtype=torch.int32)
    metadata = VarlenAttentionMetadata(
        cu_seq_q=offsets, cu_seq_k=offsets, max_q=4, max_k=4
    )
    return {
        "attention_metadata": {VarlenInnerAttention: metadata},
        "positions": torch.arange(4, dtype=torch.int32),
    }


def test_aot_parallel_module_forwards_keyword_inputs():
    from torchtitan.experiments.graph_trainer import autoparallel_api

    model = _MetadataModel()
    reference = copy.deepcopy(model)
    tokens = torch.randn(4, 3)
    autop = autoparallel_api.AutoParallelGraph.__new__(
        autoparallel_api.AutoParallelGraph
    )
    autop.model = model
    autop.compiler_fn = boxed_nop_preserve_node_meta
    with ExitStack() as stack:
        autop.joint_with_descriptors = aot_export_joint_with_descriptors(
            stack, model, (tokens,), _metadata_kwargs(2)
        )
        autop.gm = autop.joint_with_descriptors.graph_module
        with patch.object(
            autop,
            "_apply_placement_common",
            return_value=(dict(model.named_parameters()), dict(model.named_buffers())),
        ):
            parallel_model = autop.apply_placement_for_fx_module(
                compile_config=GraphTrainerCompileConfig()
            )

    kwargs = dict(reversed(list(_metadata_kwargs(3).items())))
    output = parallel_model(tokens, **kwargs)
    expected = reference(tokens, **kwargs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    (grad,) = torch.autograd.grad(output.sum(), parallel_model.projection.weight)
    (expected_grad,) = torch.autograd.grad(expected.sum(), reference.projection.weight)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)
