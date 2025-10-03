# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard, Replicate
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing


from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_tp
from torchtitan.tools.logging import logger

from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel, MixedPrecisionPolicy

from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._functorch.partitioners import min_cut_rematerialization_partition

from torch._dynamo.functional_export import _dynamo_graph_capture_for_export

from torch._functorch._aot_autograd.aot_eager_runner import (
    get_num_mutate_inputs,
    get_num_user_outputs,
    JointGraphModule,
    RunMode,
)
import contextlib
from torchtitan.experiments.simple_fsdp.llama3.model import SimpleFSDPTransformer
from torch._functorch.aot_autograd import (
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)
from torch._logging import trace_structured
    
# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}

def print_if_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)

def graph_capture_and_aot_export_joint_with_descriptors(model, inputs):
    assert isinstance(inputs, tuple)
    with torch._dynamo.config.patch(install_free_tensors=True):
        # TODO: switch to use the official graph_capture API once it is ready
        gm = _dynamo_graph_capture_for_export(model)(*inputs)
    return aot_export_joint_with_descriptors_alone(gm, inputs)


def aot_export_joint_with_descriptors_alone(model, inputs):
    assert isinstance(inputs, tuple)
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            inputs,
        )
        return joint_with_descriptors



def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.world_mesh["tp"]
        apply_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(job_config, tp_mesh)

    if job_config.activation_checkpoint.mode != "none":
        use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
        model_compile_enabled = (
            job_config.compile.enable and "model" in job_config.compile.components
        )
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=_op_sac_save_list,
        )

    # apply data parallel
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
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

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        model = data_parallel(
            model,
            parallel_dims.world_mesh[tuple(dp_mesh_dim_names)],
            mode=dp_mode,
            ac_mode=job_config.activation_checkpoint.mode,
            mp_policy=mp_policy,
        )
        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )

    if job_config.compile.enable and "model" in job_config.compile.components:
        model = HijackWrapper(model, parallel_dims)

    return model

# Just to bootstrap our experiment. NOT the final API.
class HijackWrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module, parallel_dims, **overrides):
        super().__init__()
        self.inner = inner           # register as submodule
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

        # print_if_rank0(self.parallel_dims.world_mesh)
        # 2-D device mesh with ['dp_shard', 'tp'], [2, 4]

        # Hack: convert args and kwargs to DTensor. This should be fixed at data loader. 
        # This works, but kinda cheating?
        dt_args = tuple(DTensor.from_local(arg, self.parallel_dims.world_mesh["tp"], [Replicate()]) for arg in args)

        # RuntimeError('Sharding propagation failed for Op(op=aten.embedding.default, args_schema=Spec(S(0) on (2048, 256)), Spec((Shard(dim=0), Replicate()) on (16, 2048)) @ mesh: (2, 4))')
        # dt_args = tuple(DTensor.from_local(arg, self.parallel_dims.world_mesh, [Shard(0), Replicate()]) for arg in args)

        # RuntimeError('Sharding propagation failed for Op(op=aten.embedding.default, args_schema=Spec(S(0) on (2048, 256)), Spec(S(0) on (16, 2048)) @ mesh: (2,))')
        # dt_args = tuple(DTensor.from_local(arg, self.parallel_dims.world_mesh["dp_shard"], [Shard(0)]) for arg in args) 

        # HACK: doing graph capture on the fly, we should do it AOT
        if self.joint_graph_module is None:
            # needed to avoid having fwd_rng_state in the fw_gm inp
            # this doesn't work!
            # torch._functorch.config.graphsafe_rng_functionalization = False

            # first time, we need to initialize the runner
            self.joint_graph_module = joint_graph_builder(self.inner, *dt_args, **kwargs)

        # calling the line below returns control to torchtitan's runner
        # letting it call the backward, and optimizer.
        # return self.joint_graph_module(*args, **kwargs)
        return self.joint_graph_module(args)


def joint_graph_builder(model, *inputs, **kwargs):
    assert isinstance(model, SimpleFSDPTransformer)
    assert isinstance(inputs, tuple)
    for input in inputs:
        assert isinstance(input, DTensor)
    assert not kwargs

    # get fw/bw graphs
    joint_with_descriptors = graph_capture_and_aot_export_joint_with_descriptors(model, inputs)

    # Now partition the joint grapg 
    joint_gm = joint_with_descriptors.graph_module
    aot_state = joint_with_descriptors._aot_state
    aot_graph_capture = joint_with_descriptors._aot_graph_capture

    # Get the joint graph module
    joint_inputs = aot_graph_capture.updated_flat_args
    fw_metadata = aot_state.fw_metadata

    num_user_outputs = get_num_user_outputs(fw_metadata)
    num_mutate_inputs = get_num_mutate_inputs(fw_metadata)
    num_inner_fwd_outputs = num_mutate_inputs + num_user_outputs

    fw_gm, bw_gm = min_cut_rematerialization_partition(
        joint_gm,
        joint_inputs,
        num_fwd_outputs=num_inner_fwd_outputs,
        static_lifetime_input_indices=fw_metadata.static_input_indices or [],
    )

    # print_if_rank0(f"fw_gm:")
    # print_if_rank0(fw_gm.print_readable(print_output=False))

    # print_if_rank0(f"bw_gm:")
    # print_if_rank0(bw_gm.print_readable(print_output=False))

    # Run graph passes here

    ## Apply bucketing
    # schedule_overlap_bucketing(fw_gm)
    # schedule_overlap_bucketing(bw_gm)

    ## TODO: Apply Flex Attention compilation here

    # Codgen Autograd.Function Wrappers

    # Get the model parameters and buffers - the partitioned graphs expect these as arguments
    local_params = []
    for p in model.parameters():
        if isinstance(p, DTensor):
            local_params.append(p.to_local())
        else:
            local_params.append(p)

    local_buffers = []
    for b in model.buffers():
        if isinstance(b, DTensor):
            local_buffers.append(b.to_local())
        else:
            local_buffers.append(b)


    joint_graph_module = JointGraphModule(
        local_params, local_buffers, fw_metadata, fw_gm, bw_gm, RunMode.CODEGEN_AUTOGRAD, f"rank{torch.distributed.get_rank()}"
    )

    return joint_graph_module


    # trace_structured(
    #     "artifact",
    #     metadata_fn=lambda: {
    #         "name": "aot_export_joint_with_descriptors",
    #         "encoding": "string",
    #     },
    #     payload_fn=lambda: gm.print_readable(
    #         print_output=False, include_stride=True, include_device=True
    #     ),
    # )
