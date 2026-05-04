# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from typing import Callable, Union

import torch
from autoparallel import AutoParallelPP
from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    dsv3_loss_fn,
    MoEArgs,
)
from autoparallel.graph_passes.graph_multiplex import multiplex_fw_bw_graph
from autoparallel.graph_passes.graph_pp_runner import (
    _run_dI_bw_module,
    _run_dW_bw_module,
    _run_full_bw_module,
    _run_fw_module,
    _run_multiplexed_fw_bw_module,
    _run_reduce_grad_module,
    _run_unshard_module,
    GraphCallables,
    GraphMeta,
)
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    unset_fake_temporarily,
)
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore


def compare_tuples(tuple1: Union[list, tuple], tuple2: Union[list, tuple]) -> bool:
    """Compare two tuples element-by-element with specialized comparison logic.

    For each element pair:
    - If both are FakeTensor: compare shape, stride, and dtype
    - If both are Tensor: use torch.allclose for numerical comparison
    - Otherwise: skip the comparison (always matches)

    Args:
        tuple1: First tuple to compare.
        tuple2: Second tuple to compare.

    Returns:
        True if all comparable elements match according to the rules above, False otherwise.
    """
    if len(tuple1) != len(tuple2):
        return False
    with unset_fake_temporarily():
        for elem1, elem2 in zip(tuple1, tuple2):
            # Check if both are FakeTensor
            if isinstance(elem1, FakeTensor):
                if not isinstance(elem2, FakeTensor):
                    return False
                if elem1.dtype != elem2.dtype:
                    return False
                # Try to compare strides or shape, but skip if it would trigger data-dependent guards
                try:
                    if elem1.shape != elem2.shape:
                        return False
                    if elem1.stride() != elem2.stride():
                        return False
                except (
                    torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
                ):
                    # Skip stride or shape comparison for symbolic shapes
                    pass
            # Check if both are regular Tensor (but not FakeTensor)
            elif isinstance(elem1, torch.Tensor) and not isinstance(elem1, FakeTensor):
                if not (
                    isinstance(elem2, torch.Tensor)
                    and not isinstance(elem2, FakeTensor)
                ):
                    return False
                # Use torch.allclose for numerical comparison
                if not torch.allclose(elem1, elem2):
                    return False
            # Otherwise, skip the comparison (neither or mismatched types)

    return True


def _extract_graph_modules_and_meta(
    res: dict,
) -> tuple[GraphCallables, GraphMeta]:
    graph_callables = res["graph_callables"]
    graph_modules = GraphCallables(
        fw=graph_callables["fw"],
        full_bw=graph_callables["full_bw"],
        bw_dI=graph_callables["bw_dI"],
        bw_dW=graph_callables["bw_dW"],
        unshard=graph_callables["unshard"],
        reduce_grad=graph_callables["reduce_grad"],
    )
    graph_meta = res["graph_meta"]
    graph_meta = GraphMeta(
        num_mutate_inputs=graph_meta["num_mutate_inputs"],
        num_user_outputs=graph_meta["num_user_outputs"],
        num_symints_saved_for_bw=graph_meta["num_symints_saved_for_bw"],
        num_params=graph_meta["num_params"],
        num_buffers=graph_meta["num_buffers"],
        num_input_grads=graph_meta["num_input_grads"],
    )
    return graph_modules, graph_meta


def _get_fw_inputs(
    pp_mod: torch.nn.Module, eval_input_fn: Callable
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    x: list[torch.Tensor] = list(eval_input_fn())
    sharded_params = [
        v.to_local() if isinstance(v, DTensor) else v
        for k, v in dict(pp_mod.named_parameters(remove_duplicate=False)).items()
    ]
    buffers = [
        v.to_local() if isinstance(v, DTensor) else v
        for k, v in dict(pp_mod.named_buffers(remove_duplicate=False)).items()
    ]
    return (sharded_params, buffers, x)


# Symbolically evaluate in case you want to test running a graph bigger than your gpu


def _run_graph_test(
    pp_mod: torch.nn.Module,
    graph_modules: GraphCallables,
    graph_meta: GraphMeta,
    sharded_params: list[torch.Tensor],
    buffers: list[torch.Tensor],
    x: list[torch.Tensor],
    fake_evaluate: bool,
    use_fsdp_collectives: bool,
    use_split_dI_dW: bool,
    use_multiplexed_graph: bool,
) -> None:
    """Execute forward and backward passes with specified graph options."""
    if use_multiplexed_graph:
        multiplexed_fw_bw_module = multiplex_fw_bw_graph(
            graph_modules.fw, graph_modules.full_bw, overlap_with_annotations=True
        )
    with (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    ):
        with torch.no_grad():
            # Forward pass setup
            if use_fsdp_collectives:
                unshard_args = list(sharded_params)
                assert graph_modules.unshard is not None
                unsharded_params = _run_unshard_module(
                    graph_modules.unshard, graph_meta, unshard_args
                )
                fw_args = [*unsharded_params, *buffers, *x]
            else:
                fw_args = [*sharded_params, *buffers, *x]
            if use_multiplexed_graph:
                m_fw_args = list(fw_args)
            # Forward pass
            loss_or_output, saved_intermediates = _run_fw_module(
                graph_modules.fw, graph_meta, fw_args
            )
            tangents = [torch.ones_like(loss_or_output)]
            tensors_for_backward, non_tensors_for_backward = saved_intermediates

            # Backward pass setup
            bw_args = [
                *non_tensors_for_backward,
                *tensors_for_backward,
                *tangents,
            ]
            if use_multiplexed_graph:
                m_bw_args = list(bw_args)
                joint_args = m_bw_args + m_fw_args
                del m_bw_args, m_fw_args
                (
                    m_input_grads,
                    m_param_buffer_grads,
                    m_loss_or_output,
                    m_saved_intermediates,
                ) = _run_multiplexed_fw_bw_module(
                    multiplexed_fw_bw_module, graph_meta, graph_meta, joint_args
                )
                (
                    m_tensors_for_backward,
                    m_non_tensors_for_backward,
                ) = m_saved_intermediates
                assert compare_tuples((m_loss_or_output,), (loss_or_output,))
                assert compare_tuples(m_tensors_for_backward, tensors_for_backward)
                assert compare_tuples(
                    m_non_tensors_for_backward, non_tensors_for_backward
                )
                del (
                    m_non_tensors_for_backward,
                    m_tensors_for_backward,
                    m_loss_or_output,
                )
            del (
                tensors_for_backward,
                non_tensors_for_backward,
                tangents,
                saved_intermediates,
            )

            # Backward pass
            if use_split_dI_dW:
                assert graph_modules.bw_dI is not None
                input_grads, activations_for_backward = _run_dI_bw_module(
                    graph_modules.bw_dI, graph_meta, bw_args
                )
                dw_args = list(activations_for_backward)
                del activations_for_backward
                assert graph_modules.bw_dW is not None
                param_buffer_grads = _run_dW_bw_module(
                    graph_modules.bw_dW, graph_meta, dw_args
                )
            else:
                input_grads, param_buffer_grads = _run_full_bw_module(
                    graph_modules.full_bw, graph_meta, bw_args
                )
            if use_multiplexed_graph:
                assert compare_tuples(m_param_buffer_grads, param_buffer_grads)
                assert compare_tuples(m_input_grads, input_grads)
                del m_param_buffer_grads, m_input_grads
            assert len(param_buffer_grads) == (len(sharded_params) + len(buffers))
            unsharded_grads = list(param_buffer_grads[: len(sharded_params)])
            del param_buffer_grads, input_grads
            # Gradient reduction (if using FSDP collectives)
            if use_fsdp_collectives:
                assert graph_modules.reduce_grad is not None
                sharded_grads = _run_reduce_grad_module(
                    graph_modules.reduce_grad, graph_meta, unsharded_grads
                )
            else:
                sharded_grads = unsharded_grads
            assert len(sharded_grads) == len(sharded_params)


def run_all_graph_pass_tests(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    tracing_input_fn: Callable,
    eval_input_fn: Callable,
    fake_evaluate: bool = True,
    use_loss_fn: bool = True,
):
    test_configs: list[tuple[str, list[str], bool, bool]] = [
        ("graph_partition", [], False, False),
        ("split_fsdp_collectives", ["split_fsdp_collectives"], True, False),
        ("split_dI_dW", ["split_dI_dW"], False, True),
        ("combined", ["split_fsdp_collectives", "split_dI_dW"], True, True),
    ]

    with AutoParallelPP(
        model,
        tracing_input_fn,
        mesh,
        dynamic=True,
        reshard_after_forward=False,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(0))
        if use_loss_fn:
            autop.add_input_constraints([x_sharding, x_sharding])
            autop.add_output_constraints([(Replicate(), Replicate())])
        else:
            autop.add_input_constraints([x_sharding])
            autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()

        # Get pp_mod and inputs once (identical across all graph_passes configs)
        res = autop.apply_placement_pp(
            sharding_placement=sharding_placement,
            graph_passes=[],
        )
        pp_mod = autop.parallel_model
        with unset_fake_temporarily():
            pp_mod.to_empty(device="cuda")
            pp_mod.init_weights(buffer_device="cuda")
            sharded_params, buffers, x = _get_fw_inputs(pp_mod, eval_input_fn)

        for name, graph_passes, use_fsdp, use_split in test_configs:
            if graph_passes:
                res = autop.apply_placement_pp(
                    sharding_placement=sharding_placement,
                    graph_passes=graph_passes,
                )
            graph_modules, graph_meta = _extract_graph_modules_and_meta(res)
            _run_graph_test(
                pp_mod,
                graph_modules,
                graph_meta,
                sharded_params,
                buffers,
                x,
                fake_evaluate,
                use_fsdp_collectives=use_fsdp,
                use_split_dI_dW=use_split,
                use_multiplexed_graph=True,
            )
            print(f"{name}: All good!")


if __name__ == "__main__":
    # must symbolically evaluate to run on 32 dp ranks
    # world_size = 2048
    fake_evaluate = True
    use_loss_fn = True

    world_size = 256

    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )
    # mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 64, 64),
        mesh_dim_names=(
            "dp",
            "ep",
        ),
    )

    device = torch.device("cuda")

    bs = 4 * mesh.shape[0] * mesh.shape[1]
    seq_len = 1024

    config = DeepSeekV3ModelArgs(
        vocab_size=102400,
        max_seq_len=seq_len,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=1,  # 27,
        n_dense_layers=0,  # 1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
            mesh=mesh,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        use_flex_attn=False,
        attn_mask_type="causal",
    )

    # parallelize the model
    with torch.device("meta"):
        model = DeepSeekV3Model(config).bfloat16()
        model.tok_embeddings = None  # type: ignore[assignment]

    if use_loss_fn:

        class ModelWithLoss(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, h, labels):
                output = self.model(h)
                return dsv3_loss_fn(output, labels)

            def init_weights(self, *args, **kwargs):
                return self.model.init_weights(*args, **kwargs)

        model = ModelWithLoss(model)

    def make_input_fn(sharded: bool = False, with_target: bool = False):
        """Create input generator. `sharded` uses mesh-adjusted batch size."""

        def input_fn() -> tuple[torch.Tensor, ...]:
            batch_size = bs // (mesh.shape[0] * mesh.shape[1]) if sharded else bs

            inputs = (
                torch.randn(
                    (batch_size, seq_len, config.dim),
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                ),
            )
            if with_target:
                inputs += (
                    torch.randint(
                        0, config.vocab_size, (batch_size, seq_len), device=device
                    ),
                )
            return inputs

        return input_fn

    input_fn = make_input_fn(sharded=False, with_target=use_loss_fn)
    eval_fn = make_input_fn(sharded=True, with_target=use_loss_fn)

    run_all_graph_pass_tests(model, mesh, input_fn, eval_fn, fake_evaluate, use_loss_fn)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
