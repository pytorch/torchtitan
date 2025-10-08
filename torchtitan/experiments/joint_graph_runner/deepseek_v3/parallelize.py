# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch
import torch.nn as nn

from torch._dynamo.functional_export import _dynamo_graph_capture_for_export

from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch._guards import tracing, TracingContext
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp

from torchtitan.experiments.simple_fsdp.deepseek_v3.model import (
    SimpleFSDPDeepSeekV3Model,
)
from torchtitan.experiments.simple_fsdp.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)

from torchtitan.models.deepseek_v3.infra.parallelize import (
    apply_ac,
    apply_moe_ep_tp,
    apply_non_moe_tp,
)
from torchtitan.tools.logging import logger


def print_if_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def graph_capture_and_aot_export_joint_with_descriptors(model, inputs):
    assert isinstance(inputs, tuple)
    with torch._dynamo.config.patch(
        install_free_tensors=True
    ), torch.fx.traceback.preserve_node_meta():
        # TODO: switch to use the official graph_capture API once it is ready
        gm = _dynamo_graph_capture_for_export(model)(*inputs)

        # Restore the state dict to match the original module
        _restore_state_dict(model, gm)

        print_if_rank0("Dynamo gm:")
        print_if_rank0(gm.print_readable(print_output=False))

        fake_mode = gm.meta.get("fake_mode", None)

    with tracing(TracingContext(fake_mode)):
        return aot_export_joint_with_descriptors_alone(gm, inputs), fake_mode


def aot_export_joint_with_descriptors_alone(model, inputs):
    assert isinstance(inputs, tuple)
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            inputs,
        )
        return joint_with_descriptors


def _clear_traced_params_buffers(
    traced_module: torch.fx.GraphModule, const_keys: list[str]
) -> None:
    """Remove all parameters and buffers from traced module before restoring."""
    for key in const_keys:
        assert key in traced_module._buffers.keys()
        # We don't want constants to show up as a buffer in the state dict.
        # Instead they should just be a direct attribute.
        buffer = getattr(traced_module, key)
        torch.fx.graph_module._del_attr(traced_module, key)
        setattr(traced_module, key, buffer)


def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    TODO: move this into torch.export
    Restores the state dict of the traced module to match the original module exactly.
    Preserves the original FQNs with dots, creating intermediate empty modules as needed.
    Ensures that the ordering of parameters/buffers matches the original module.
    """
    # Build ID-based lookups for traced module params/buffers
    traced_params: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for name, param in traced_module.named_parameters(remove_duplicate=False):
        traced_params[id(param)] = (name, param)

    traced_buffers: dict[int, tuple[str, torch.Tensor]] = {}
    for name, buffer in traced_module.named_buffers(remove_duplicate=False):
        traced_buffers[id(buffer)] = (name, buffer)

    # Build mapping from old names to new names for graph node updates
    name_mapping: dict[str, str] = {}

    # Restore parameters in the order they appear in original module
    for orig_name, orig_param in original_module.named_parameters(
        remove_duplicate=False
    ):
        if id(orig_param) in traced_params:
            # This param exists in traced module - restore it with original FQN
            traced_name, traced_param = traced_params[id(orig_param)]
            torch.fx.graph_module._assign_attr(traced_param, traced_module, orig_name)
            torch.fx.graph_module._del_attr(traced_module, traced_name)
            name_mapping[traced_name] = orig_name
        else:
            # This param doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_param, traced_module, orig_name)

    # Restore buffers in the order they appear in original module
    for orig_name, orig_buffer in original_module.named_buffers(remove_duplicate=False):
        if id(orig_buffer) in traced_buffers:
            # This buffer exists in traced module - restore it with original FQN
            traced_name, traced_buffer = traced_buffers[id(orig_buffer)]
            torch.fx.graph_module._assign_attr(orig_buffer, traced_module, orig_name)
            name_mapping[traced_name] = orig_name
            torch.fx.graph_module._del_attr(traced_module, traced_name)
        else:
            # This buffer doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_buffer, traced_module, orig_name)

    param_names = [v[0] for v in traced_params.values()]
    buffer_names = [v[0] for v in traced_buffers.values()]
    const_keys = set(param_names + buffer_names).difference(set(name_mapping.keys()))

    _clear_traced_params_buffers(traced_module, const_keys)

    # Update get_attr nodes in the graph to use the correct FQNs
    for node in traced_module.graph.nodes:
        if node.op == "get_attr" and node.target in name_mapping:
            node.target = name_mapping[node.target]

    traced_module.recompile()


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    world_mesh = parallel_dims.world_mesh
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """
    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.use_flex_attn
    ):
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            use_flex_attn=use_flex_attn,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
        )
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )

    # apply data parallel
    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.fsdp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.dp_replicate_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ("dp_replicate",)
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
            dp_mode = "fully_shard"
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")
        dp_mod_ep_mesh = world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
        for _, transformer_block in model.layers.items():
            if transformer_block.moe_enabled and parallel_dims.ep_enabled:
                experts_shard_dim = 0
                assert dp_mod_ep_mesh is not None
                assert hasattr(transformer_block, "moe")
                if (
                    dp_mod_ep_mesh.size() * parallel_dims.ep
                    > transformer_block.moe.experts.num_experts
                ):
                    experts_shard_dim = 1
                transformer_block.moe.experts = data_parallel(
                    transformer_block.moe.experts,
                    dp_mod_ep_mesh,
                    dp_mode,
                    ac_mode=job_config.activation_checkpoint.mode,
                    mp_policy=mp_policy,
                    shard_dim=experts_shard_dim,
                )
                # TODO(ruisizhang123): support set_gradient_divide_factor in simplefsdp
                # transformer_block.moe.experts.set_gradient_divide_factor(
                #     parallel_dims.fsdp_gradient_divide_factor,
                # )
        model = data_parallel(
            model,
            dp_mesh,
            dp_mode,
            ac_mode=job_config.activation_checkpoint.mode,
            mp_policy=mp_policy,
        )
        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )
    if job_config.compile.enable:
        model = HijackWrapper(model, parallel_dims)
    return model


class HijackWrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module, parallel_dims, **overrides):
        super().__init__()
        self.inner = inner  # register as submodule
        self.parallel_dims = parallel_dims

        self.joint_graph_module = None
        self._overrides = overrides  # for custom hooks

    def __getattr__(self, name):
        # check overrides
        if "_overrides" in self.__dict__ and name in self._overrides:
            return self._overrides[name]
        try:
            # let nn.Module handle registered stuff
            return super().__getattr__(name)
        except AttributeError:
            # fallback to inner model
            return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if "_overrides" in self.__dict__ and name in self._overrides:
            self._overrides[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if "_overrides" in self.__dict__ and name in self._overrides:
            del self._overrides[name]
        else:
            super().__delattr__(name)

    def forward(self, *args, **kwargs):
        assert "forward" not in self._overrides, "forward cannot be overridden"
        dt_args = tuple(
            DTensor.from_local(arg, self.parallel_dims.world_mesh["tp"], [Replicate()])
            for arg in args
        )
        if self.joint_graph_module is None:
            self.joint_graph_module = joint_graph_builder(
                self.inner, *dt_args, **kwargs
            )

        # calling the line below returns control to torchtitan's runner
        # letting it call the backward, and optimizer.
        # return self.joint_graph_module(*args, **kwargs)
        return self.joint_graph_module(args)


def joint_graph_builder(model, *inputs, **kwargs):
    assert isinstance(model, SimpleFSDPDeepSeekV3Model)
    assert isinstance(inputs, tuple)
    for input in inputs:
        assert isinstance(input, DTensor)
    assert not kwargs

    # get fw/bw graphs
    (
        joint_with_descriptors,
        fake_mode,
    ) = graph_capture_and_aot_export_joint_with_descriptors(model, inputs)

    # verify user annotation show up in the graph
    for node in joint_with_descriptors.graph_module.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            if "custom" not in node.meta:
                # this is currently failing, as backward nodes are missing the annotation
                # raise RuntimeError(f"node {node} is not annotated with custom metadata, seeing node.meta: {node.meta}")
                pass

    def compiler(gm: torch.fx.GraphModule, example_inputs):
        print_if_rank0("Before compiler:")
        print_if_rank0(gm.print_readable(print_output=False))

        print_if_rank0("After compiler:")
        print_if_rank0(gm.print_readable(print_output=False))
        return gm

    with tracing(TracingContext(fake_mode)):
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors, fw_compiler=compiler, bw_compiler=compiler
        )

    def wrapper_fn(args):
        input = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        return fn(*input)

    return wrapper_fn
