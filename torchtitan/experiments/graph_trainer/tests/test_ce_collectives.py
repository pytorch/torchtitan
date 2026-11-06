# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig
from torchtitan.experiments.graph_trainer.ce_collectives import ce_collectives_pass
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.passes import compile_time_passes
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META

c10d = torch.ops._c10d_functional
ce = torch.ops.graph_trainer_ce


def _graph(kind, *, fsdp=True, group_size=2, reduce_op="sum"):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    out = graph.placeholder("out")
    if kind == "out":
        node = graph.call_function(
            c10d.all_gather_into_tensor_out.default,
            (source, group_size, "fsdp"),
            {"out": out},
        )
    elif kind == "all_gather":
        node = graph.call_function(
            c10d.all_gather_into_tensor.default, (source, group_size, "fsdp")
        )
    else:
        node = graph.call_function(
            c10d.reduce_scatter_tensor.default,
            (source, reduce_op, group_size, "fsdp"),
        )
    if fsdp:
        node.meta["custom"] = {FSDP_PARAM_FQNS_META: ("layers.0.attention.wq.weight",)}
    wait = graph.call_function(c10d.wait_tensor.default, (node,))
    graph.output((wait, out))
    return torch.fx.GraphModule({}, graph), node, wait, out


@pytest.mark.parametrize("kind", ["all_gather", "out", "reduce_scatter"])
@pytest.mark.parametrize("workspace_size_mib", [10, 32])
def test_rewrite_preserves_wait_and_output_storage(kind, workspace_size_mib):
    gm, node, wait, out = _graph(kind)
    ce_collectives_pass(
        gm, config=CECollectivesConfig(), workspace_size_mib=workspace_size_mib
    )
    expected = {
        "all_gather": ce.all_gather_into_tensor.default,
        "out": ce.all_gather_into_tensor_out.default,
        "reduce_scatter": ce.reduce_scatter_tensor.default,
    }
    assert node.target is expected[kind]
    assert node.kwargs["workspace_size_mib"] == workspace_size_mib
    assert wait.target is c10d.wait_tensor.default
    assert wait.args[0] is (out if kind == "out" else node)
    if kind == "out":
        assert node.kwargs["out"] is out
        gm.graph.eliminate_dead_code()
        assert node in gm.graph.nodes
    gm.graph.lint()


def test_invalid_workspace_size_is_rejected():
    gm, _, _, _ = _graph("all_gather")
    with pytest.raises(ValueError, match="workspace_size_mib must be positive"):
        ce_collectives_pass(gm, config=CECollectivesConfig(), workspace_size_mib=0)


@pytest.mark.parametrize("kind", ["all_gather", "out", "reduce_scatter"])
def test_model_parallel_collectives_are_unchanged(kind):
    gm, node, _, _ = _graph(kind, fsdp=False)
    target = node.target
    ce_collectives_pass(gm, config=CECollectivesConfig())
    assert node.target is target


def test_mixed_provenance_bucket_is_unchanged():
    gm, node, _, _ = _graph("out")
    node.meta["bucketing_custom_sources"] = [node.meta["custom"], {}]
    ce_collectives_pass(gm, config=CECollectivesConfig())
    assert node.target is c10d.all_gather_into_tensor_out.default


def test_dense_scope_preserves_any_bucket_containing_excluded_parameters():
    gm, node, _, _ = _graph("out")
    node.meta["bucketing_custom_sources"] = [
        node.meta["custom"],
        {FSDP_PARAM_FQNS_META: ("layers.1.feed_forward.experts.weight",)},
    ]
    ce_collectives_pass(
        gm, config=CECollectivesConfig(), excluded_module_fqns=("layers.1",)
    )
    assert node.target is c10d.all_gather_into_tensor_out.default


def test_all_gather_requires_one_domain():
    gm, _, _, _ = _graph("all_gather", group_size=4)
    with pytest.raises(ValueError, match="one NVLink domain"):
        ce_collectives_pass(gm, config=CECollectivesConfig(ranks_per_domain=2))


def test_hierarchical_reduce_scatter_and_disabled_all_gather():
    gm, node, _, _ = _graph("reduce_scatter", group_size=4, reduce_op="avg")
    ce_collectives_pass(
        gm, config=CECollectivesConfig(all_gather=False, ranks_per_domain=2)
    )
    assert node.target is ce.reduce_scatter_tensor.default
    assert node.kwargs["ranks_per_domain"] == 2
    assert node.kwargs["reduce_op"] == "avg"
    gm, node, _, _ = _graph("out")
    ce_collectives_pass(gm, config=CECollectivesConfig(all_gather=False))
    assert node.target is c10d.all_gather_into_tensor_out.default


def test_unsupported_reduction_is_rejected():
    gm, _, _, _ = _graph("reduce_scatter", reduce_op="max")
    with pytest.raises(ValueError, match="SUM and AVG"):
        ce_collectives_pass(gm, config=CECollectivesConfig())


def test_meta_shapes_and_mutable_output():
    source = torch.empty(4, 3, device="meta")
    gathered = ce.all_gather_into_tensor(source, 2, "fsdp")
    assert gathered.shape == (8, 3)
    assert ce.all_gather_into_tensor_out(source, 2, "fsdp", out=gathered) is None
    reduced = ce.reduce_scatter_tensor(source, "sum", 2, "fsdp", ranks_per_domain=2)
    assert reduced.shape == (2, 3)


def _passes(compile_config):
    return compile_time_passes(
        SimpleNamespace(state_fqns=[]),
        SimpleNamespace(
            compile=compile_config,
            model=SimpleNamespace(layers=[]),
            parallelism=SimpleNamespace(),
            training=SimpleNamespace(disable_accelerator_graphs=True),
        ),
    )


def test_opt_in_pass_runs_after_bucketing_before_inductor():
    default = [
        getattr(p, "func", p).__name__ for p in _passes(GraphTrainerCompileConfig())
    ]
    assert "ce_collectives_pass" not in default
    enabled = [
        getattr(p, "func", p).__name__
        for p in _passes(
            GraphTrainerCompileConfig(
                ce_collectives=CECollectivesConfig(), numerics_changing_optim=True
            )
        )
    ]
    assert enabled.index(
        "joint_transformer_block_bucketing_reordering_pass"
    ) < enabled.index("ce_collectives_pass")
    assert enabled.index("ce_collectives_pass") < enabled.index(
        "regional_inductor_pass"
    )


def test_reduce_scatter_requires_numerics_opt_in():
    with pytest.raises(ValueError, match="numerics_changing_optim"):
        _passes(GraphTrainerCompileConfig(ce_collectives=CECollectivesConfig()))
    _passes(
        GraphTrainerCompileConfig(
            ce_collectives=CECollectivesConfig(reduce_scatter=False)
        )
    )
