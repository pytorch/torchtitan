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

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.configs import (
    AUTOPARALLEL_BACKEND,
    GraphTrainerCompileConfig,
    is_autoparallel_backend_mode,
    validate_autoparallel_backend_config,
)
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN


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
        self.used_backend_path = False
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
        return torch.nn.Linear(1, 1)


def _training_config():
    return TrainingConfig(
        local_batch_size=2,
        seq_len=8,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def _single_override_args(test_definition):
    [args] = test_definition.override_args
    return list(args)


def test_autoparallel_integration_matrix_is_paired():
    from torchtitan.experiments.graph_trainer.tests.integration_tests import (
        build_graph_trainer_autoparallel_test_list,
    )

    tests = build_graph_trainer_autoparallel_test_list()
    tests_by_name = {test.test_name: test for test in tests}

    assert set(tests_by_name) == {
        "autoparallel_llama3_fsdp_tp",
        "autoparallel_backend_llama3_fsdp_tp",
        "autoparallel_deepseek_v3_efsdp_ep",
        "autoparallel_backend_deepseek_v3_efsdp_ep",
    }
    assert len(tests) == 4

    for native_name, backend_name in [
        ("autoparallel_llama3_fsdp_tp", "autoparallel_backend_llama3_fsdp_tp"),
        (
            "autoparallel_deepseek_v3_efsdp_ep",
            "autoparallel_backend_deepseek_v3_efsdp_ep",
        ),
    ]:
        native = tests_by_name[native_name]
        backend = tests_by_name[backend_name]
        native_args = _single_override_args(native)
        backend_args = _single_override_args(backend)

        assert native.ngpu == backend.ngpu == 4
        backend_flag = "--compile.inductor_compilation autoparallel_backend"
        assert backend_flag not in native_args
        assert backend_flag in backend_args
        backend_args_without_backend = [
            arg for arg in backend_args if arg != backend_flag
        ]
        assert native_args == backend_args_without_backend


def test_autoparallel_backend_config_validation():
    with pytest.raises(ValueError, match="requires --compile.autoparallel"):
        validate_autoparallel_backend_config(
            GraphTrainerCompileConfig(
                inductor_compilation=AUTOPARALLEL_BACKEND,
                autoparallel=False,
            )
        )

    invalid_options = [
        GraphTrainerCompileConfig(
            inductor_compilation=AUTOPARALLEL_BACKEND,
            autoparallel=True,
            enable_passes=False,
        ),
        GraphTrainerCompileConfig(
            inductor_compilation=AUTOPARALLEL_BACKEND,
            autoparallel=True,
            passes=["regional_inductor"],
        ),
        GraphTrainerCompileConfig(
            inductor_compilation=AUTOPARALLEL_BACKEND,
            autoparallel=True,
            joint_passes=["foo"],
        ),
        GraphTrainerCompileConfig(
            inductor_compilation=AUTOPARALLEL_BACKEND,
            autoparallel=True,
            precompile_artifact_dir="/tmp/artifact",
        ),
    ]
    for compile_config in invalid_options:
        with pytest.raises(ValueError):
            validate_autoparallel_backend_config(compile_config)

    compile_config = GraphTrainerCompileConfig(
        inductor_compilation=AUTOPARALLEL_BACKEND,
        autoparallel=True,
    )
    validate_autoparallel_backend_config(compile_config)
    assert is_autoparallel_backend_mode(compile_config)


def test_autoparallel_backend_pass_selection():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            inductor_compilation=AUTOPARALLEL_BACKEND,
            autoparallel=True,
        )
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)

    assert graph_passes == [
        passes.remove_detach_pass,
        passes.remove_identity_view_pass,
        passes.remove_identity_slice_pass,
        passes.normalize_view_ops_as_reshape,
        passes.autoparallel_backend_full_inductor_compilation_pass,
    ]

    native_config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            autoparallel=True,
            enable_cudagraph=False,
        ),
        model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        parallelism=SimpleNamespace(
            enable_async_tensor_parallel=False,
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )
    graph_passes = passes.construct_default_graph_passes(traced_result, native_config)

    assert passes.autoparallel_backend_full_inductor_compilation_pass not in graph_passes


def test_autoparallel_backend_full_inductor_pass_applies_policy():
    from torchtitan.experiments.graph_trainer import passes

    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    gm = torch.fx.symbolic_trace(TestModule())
    example_inputs = (torch.ones(1),)
    joint_pass_called = False

    def joint_pass(gm, joint_inputs):
        nonlocal joint_pass_called
        joint_pass_called = True
        assert joint_inputs is example_inputs
        return gm

    def output_code(args):
        return args[0] + 2

    with (
        patch(
            "autoparallel.compile.get_autoparallel_backend_policy_helpers",
            return_value=({"joint_custom_pass": joint_pass}, {}),
        ) as policy_helper,
        patch.object(
            passes, "compile_fx_inner", return_value=output_code
        ) as compile_fx_inner,
    ):
        compiled_gm = passes.autoparallel_backend_full_inductor_compilation_pass(
            gm, example_inputs
        )

    assert compiled_gm(torch.ones(1)).item() == 3
    assert joint_pass_called
    policy_helper.assert_called_once_with()
    compile_fx_inner.assert_called_once_with(gm, example_inputs)


@pytest.mark.parametrize(
    (
        "model_name",
        "inductor_compilation",
        "fsdp_reshard_after_forward",
        "expected_reshard_after_forward",
    ),
    [
        ("llama", "regional", "always", True),
        ("deepseek", AUTOPARALLEL_BACKEND, "never", False),
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
        autoparallel=True,
        inductor_compilation=inductor_compilation,
    )
    parallelism = ParallelismConfig(
        fsdp_reshard_after_forward=fsdp_reshard_after_forward
    )

    if model_name == "llama":
        from torchtitan.experiments.graph_trainer.llama3 import (
            parallelize_autoparallel,
        )

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
            loss_fn=object(),
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
    assert autop.used_fx_path
    assert not autop.used_backend_path


def test_backward_metadata_copy_prefers_annotated_forward_node_for_shared_seq_nr():
    from torchtitan.experiments.graph_trainer.make_fx_tracer import (
        _copy_fwd_metadata_to_bw_nodes,
    )

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    unannotated_fwd = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    annotated_fwd = graph.call_function(
        torch.ops.aten.relu.default, args=(unannotated_fwd,)
    )
    bwd = graph.call_function(torch.ops.aten.neg.default, args=(annotated_fwd,))
    graph.output(bwd)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    unannotated_fwd.meta["seq_nr"] = 7
    annotated_fwd.meta["seq_nr"] = 7
    annotated_fwd.meta["custom"] = {_MODULE_FQN: "layers.0.attention"}
    bwd.meta["seq_nr"] = 7
    bwd.meta["autograd_backward"] = True

    _copy_fwd_metadata_to_bw_nodes(gm)

    assert bwd.meta["custom"][_MODULE_FQN] == "layers.0.attention"


def test_autoparallel_nested_backward_compiler_preserves_meta_and_backward_tag():
    from torch.fx.experimental.proxy_tensor import make_fx

    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        _make_graph_trainer_compilers,
    )

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, 2))
    mul.meta["custom"] = {_MODULE_FQN: "layers.0.mlp"}
    mul.meta["autograd_backward"] = True
    graph.output(mul)
    backward_gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    _, bw_compiler = _make_graph_trainer_compilers()
    compiled_backward = bw_compiler(backward_gm, (torch.ones(2),))

    traced = make_fx(lambda x: compiled_backward([x]))(torch.ones(2))
    [mul_node] = [
        node
        for node in traced.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor
    ]

    assert mul_node.meta["autograd_backward"] is True
    assert mul_node.meta["custom"][_MODULE_FQN] == "layers.0.mlp"
