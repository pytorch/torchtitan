# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
        self.mesh_dim_names = tuple(mesh_axis_names)
        self.ndim = len(self.mesh_dim_names)
        self._size = size

    def size(self):
        return self._size


class _FakeParallelDims:
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
        enabled = {"fsdp", "tp"} if not self.sparse else {"efsdp", "ep"}
        return _FakeMesh((name,)) if name in enabled else None

    def get_mesh(self, names):
        if isinstance(names, str):
            return _FakeMesh((names,))
        return _FakeMesh(tuple(names))


class _FakeAutoParallelGraph:
    instances = []

    def __init__(self, model, input_fn, mesh, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.used_fx_path = False
        _FakeAutoParallelGraph.instances.append(self)

    def __enter__(self):
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
        local_batch_size=2,
        seq_len=8,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def _build_autoparallel_a2a_linear_graph(
    *,
    with_backward=False,
    a2a_as_weight=False,
    linear_op=torch.ops.aten.mm.default,
):
    """Build the A2A -> WO -> residual shape from the real Llama graph."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")

    embedding_a2a = graph.call_function(
        torch.ops._dtensor.shard_dim_alltoall.default,
        args=(x, 2, 0, "tp"),
    )
    embedding_output_a2a = graph.call_function(
        torch.ops._dtensor.shard_dim_alltoall.default,
        args=(embedding_a2a, 2, 1, "dp"),
    )
    qkv = graph.call_function(
        torch.ops.aten.mm.default, args=(embedding_output_a2a, weight)
    )
    a2a = graph.call_function(
        torch.ops._dtensor.shard_dim_alltoall.default,
        args=(qkv, 2, 1, "tp"),
    )
    pre_wo_unsqueeze = graph.call_function(
        torch.ops.aten.unsqueeze.default,
        args=(a2a, 3),
    )
    pre_wo_permute = graph.call_function(
        torch.ops.aten.permute.default,
        args=(pre_wo_unsqueeze, [0, 1, 3, 2]),
    )
    pre_wo_reshape = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(pre_wo_permute, [1, 1024, 4096]),
    )
    pre_wo = graph.call_function(
        torch.ops.aten.squeeze.dim,
        args=(pre_wo_reshape, 0),
    )
    wo_args = (x, pre_wo) if a2a_as_weight else (pre_wo, weight)
    wo = graph.call_function(linear_op, args=wo_args)
    post_wo_unsqueeze = graph.call_function(
        torch.ops.aten.unsqueeze.default,
        args=(wo, 0),
    )
    post_wo_reshape = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(post_wo_unsqueeze, [2, 512, 1, 4096]),
    )
    post_wo_permute = graph.call_function(
        torch.ops.aten.permute.default,
        args=(post_wo_reshape, [0, 1, 3, 2]),
    )
    post_wo = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(post_wo_permute, [2, 512, 4096]),
    )
    residual_add = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(embedding_output_a2a, post_wo),
    )
    ffn_w1 = graph.call_function(
        torch.ops.aten.mm.default,
        args=(residual_add, weight),
    )
    ffn_w3 = graph.call_function(torch.ops.aten.mm.default, args=(ffn_w1, weight))
    ffn_w2 = graph.call_function(torch.ops.aten.mm.default, args=(ffn_w3, weight))

    output = ffn_w2
    if with_backward:
        backward = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(residual_add, 2)
        )
        backward.meta["autograd_backward"] = True
        output = (output, backward)
    graph.output(output)

    fqns = {
        embedding_a2a: "tok_embeddings",
        embedding_output_a2a: "tok_embeddings",
        qkv: "layers.0.attention.qkv_linear.wqkv",
        a2a: "layers.0.attention.wo",
        pre_wo_unsqueeze: "layers.0.attention.wo",
        pre_wo_permute: "layers.0.attention.wo",
        pre_wo_reshape: "layers.0.attention.wo",
        pre_wo: "layers.0.attention.wo",
        wo: "layers.0.attention.wo",
        post_wo_unsqueeze: "layers.0.attention.wo",
        post_wo_reshape: "layers.0.attention.wo",
        post_wo_permute: "layers.0.attention.wo",
        post_wo: "layers.0.attention.wo",
        residual_add: "layers.0",
        ffn_w1: "layers.0.feed_forward.w1",
        ffn_w3: "layers.0.feed_forward.w3",
        ffn_w2: "layers.0.feed_forward.w2",
    }
    for node, fqn in fqns.items():
        node.meta["custom"] = {_MODULE_FQN: fqn}

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, SimpleNamespace(
        embedding_a2a=embedding_a2a,
        embedding_output_a2a=embedding_output_a2a,
        qkv=qkv,
        a2a=a2a,
        pre_wo=pre_wo,
        wo=wo,
        post_wo=post_wo,
        residual_add=residual_add,
        ffn_w1=ffn_w1,
        ffn_w3=ffn_w3,
        ffn_w2=ffn_w2,
    )


def test_autoparallel_integration_matrix():
    from torchtitan.experiments.graph_trainer.tests.integration_tests import (
        build_graph_trainer_autoparallel_h100_test_list,
        build_graph_trainer_autoparallel_test_list,
    )

    suites = {
        "default": build_graph_trainer_autoparallel_test_list(),
        "h100": build_graph_trainer_autoparallel_h100_test_list(),
    }

    assert [test.test_name for test in suites["default"]] == [
        "autoparallel_llama3_fsdp_tp"
    ]
    assert [test.test_name for test in suites["h100"]] == [
        "autoparallel_deepseek_v3_efsdp_ep"
    ]
    assert all(test.ngpu == 4 for tests in suites.values() for test in tests)


def test_autoparallel_config_validation():
    with pytest.raises(ValueError, match="only supports --compile.mode aot_fx_trace"):
        validate_autoparallel_config(
            GraphTrainerCompileConfig(
                mode="jit",
                enable_autoparallel=True,
            )
        )

    compile_config = GraphTrainerCompileConfig(
        inductor_compilation="regional",
        enable_autoparallel=True,
    )
    validate_autoparallel_config(compile_config)


def test_autoparallel_graph_pass_selection_uses_regular_memory_policy():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            disable_passes=["cudagraph_pass"],
        ),
        model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        parallelism=SimpleNamespace(
            enable_async_tensor_parallel=False,
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass in pass_fns


@pytest.mark.parametrize(
    "linear_op", [torch.ops.aten.mm.default, torch.ops.aten.linear.default]
)
def test_autoparallel_eager_sac_saves_a2a_linear_boundaries(linear_op):
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(linear_op=linear_op)
    config = SimpleNamespace(
        compile=SimpleNamespace(enable_autoparallel=True, memory_policy="eager")
    )

    tag_with_memory_policy_pass(gm, config=config)

    assert nodes.a2a.meta["recompute"] is CheckpointPolicy.MUST_SAVE
    assert nodes.post_wo.meta["recompute"] is CheckpointPolicy.MUST_SAVE
    assert nodes.embedding_a2a.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert nodes.embedding_output_a2a.meta["recompute"] is CheckpointPolicy.MUST_SAVE
    assert nodes.pre_wo.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert [
        node.meta["recompute"]
        for node in (nodes.qkv, nodes.wo, nodes.ffn_w1, nodes.ffn_w3, nodes.ffn_w2)
    ] == [
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_SAVE,
    ]


def test_autoparallel_eager_sac_requires_a2a_as_linear_activation():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )

    gm, nodes = _build_autoparallel_a2a_linear_graph(a2a_as_weight=True)
    config = SimpleNamespace(
        compile=SimpleNamespace(enable_autoparallel=True, memory_policy="eager")
    )

    tag_with_memory_policy_pass(gm, config=config)

    assert nodes.a2a.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE
    assert nodes.post_wo.meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE


def test_autoparallel_eager_sac_does_not_rematerialize_a2a_linear():
    from torchtitan.experiments.graph_trainer.memory_policy import (
        tag_with_memory_policy_pass,
    )
    from torchtitan.experiments.graph_trainer.selective_activation_remat import (
        selective_activation_remat_pass,
    )

    def apply_passes(*, enable_autoparallel):
        gm, nodes = _build_autoparallel_a2a_linear_graph(with_backward=True)
        config = SimpleNamespace(
            compile=SimpleNamespace(
                enable_autoparallel=enable_autoparallel,
                memory_policy="eager",
            )
        )
        tag_with_memory_policy_pass(gm, config=config)
        selective_activation_remat_pass(gm)
        gm.graph.lint()
        return gm, nodes

    regular_gm, regular_nodes = apply_passes(enable_autoparallel=False)
    autoparallel_gm, autoparallel_nodes = apply_passes(enable_autoparallel=True)

    regular_names = {node.name for node in regular_gm.graph.nodes}
    autoparallel_names = {node.name for node in autoparallel_gm.graph.nodes}
    assert regular_nodes.a2a.name + "_recomputed" in regular_names
    assert regular_nodes.wo.name + "_recomputed" in regular_names
    assert regular_nodes.post_wo.name + "_recomputed" in regular_names
    assert autoparallel_nodes.a2a.name + "_recomputed" not in autoparallel_names
    assert autoparallel_nodes.wo.name + "_recomputed" not in autoparallel_names
    assert autoparallel_nodes.post_wo.name + "_recomputed" not in autoparallel_names
    assert (
        sum(
            node.target is torch.ops._dtensor.shard_dim_alltoall.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in regular_gm.graph.nodes
        )
        == 2
    )
    assert (
        sum(
            node.target is torch.ops._dtensor.shard_dim_alltoall.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in autoparallel_gm.graph.nodes
        )
        == 1
    )
    assert (
        sum(
            node.target is torch.ops.aten.mm.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in regular_gm.graph.nodes
        )
        == 2
    )
    assert (
        sum(
            node.target is torch.ops.aten.mm.default
            and node.meta.get("custom", {}).get(_MODULE_FQN) == "layers.0.attention.wo"
            for node in autoparallel_gm.graph.nodes
        )
        == 1
    )


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

        model = SimpleNamespace(config=SimpleNamespace(vocab_size=16))
        parallel_dims = _FakeParallelDims()
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_llama
        extra_patches = ()
    else:
        from torchtitan.experiments.graph_trainer.deepseek_v3 import (
            parallelize_autoparallel,
        )

        model = SimpleNamespace(config=SimpleNamespace())
        parallel_dims = _FakeParallelDims(sparse=True)
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
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)

        call_parallelize(
            model,
            parallel_dims=parallel_dims,
            training=_training_config(),
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=object(),
            dump_folder="",
        )

    autop = _FakeAutoParallelGraph.instances[0]
    mp_policy = autop.kwargs["mp_policy"]
    assert mp_policy.param_dtype is torch.bfloat16
    assert mp_policy.reduce_dtype is torch.float32
    assert autop.kwargs["reshard_after_forward"] is expected_reshard_after_forward
    assert autop.kwargs.get("dynamic", False) is (model_name == "deepseek")
    assert autop.apply_kwargs["compile_config"] is compile_config
    assert autop.used_fx_path
