# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel-based parallelization for Llama3.

Uses AutoParallelGraph to apply solver-based SPMD sharding, then lets
graph_trainer trace and compile the placed model through its normal
`aot_fx_trace` train-step pipeline.
"""

import logging
import time

import torch
from autoparallel import ForwardInputs, UNCONSTRAINED
from torch._functorch._aot_autograd.fx_utils import get_plain_input_and_grad_nodes
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.tools.utils import device_type


logger = logging.getLogger(__name__)


def parallelize_autoparallel_llama(
    model,
    *,
    parallelism_context: ParallelismContext,
    precision: PrecisionConfig,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    activation_remat: ActivationRematConfig | None,
    dump_folder: str,
    max_num_documents: int | None = None,
):
    """Apply AutoParallelGraph SPMD sharding to Llama3.

    Trace the same named inputs and fixed-capacity metadata as the dataloader.
    Metadata-backed models currently require data parallel degree one because
    sharding packed-document offsets is not supported by this adapter.
    """
    if parallelism_context.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP yet")
    if parallelism_context.cp_enabled:
        raise ValueError("AutoParallel Llama3 does not support CP yet")
    if parallelism_context.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")
    if max_num_documents is None:
        raise ValueError(
            "AutoParallel Llama3 requires a fixed dataloader max_num_documents "
            "to trace attention metadata."
        )

    # CP is rejected above, so the former flattened ``fsdp = dp_shard * cp``
    # axis maps exactly to ``dp_shard`` here.
    dense_names = ["dp_replicate", "dp_shard", "tp"]
    dense_names = [
        name
        for name in dense_names
        if parallelism_context.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallelism_context.get_mesh(dense_names)

    dp_degree = parallelism_context.dp_replicate * parallelism_context.dp_shard
    num_tokens = training.num_tokens_per_microbatch_per_dp_rank * dp_degree
    traced_inputs: ForwardInputs | None = None

    def input_fn():
        nonlocal traced_inputs
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (num_tokens,),
            device=torch.device(device_type),
        )
        positions = (
            torch.arange(
                num_tokens,
                dtype=torch.int32,
                device=torch.device(device_type),
            )
            % model.config.max_context_length
        )
        padding_mask = torch.zeros_like(tokens, dtype=torch.bool)
        attention_metadata = model._get_attention_metadata(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
        )
        if attention_metadata and dp_degree != 1:
            raise ValueError(
                "AutoParallel Llama3 attention metadata currently requires "
                "data parallel degree one; packed-document offsets cannot be "
                "sharded across data-parallel ranks."
            )
        traced_inputs = ForwardInputs(
            args=(tokens,),
            kwargs={
                "positions": positions,
                "padding_mask": padding_mask,
                "attention_metadata": attention_metadata,
            },
        )
        return traced_inputs

    param_dtype = TORCH_DTYPE_MAP[precision.param_dtype]
    reduce_dtype = TORCH_DTYPE_MAP[precision.reduce_dtype]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallelism_context.pp_enabled,
    )

    possible_input_shardings = {
        "dp_replicate": Shard(0),
        "dp_shard": Shard(0),
        "tp": Replicate(),
    }
    unsupported_axes = [
        name
        for name in dense_mesh.mesh_dim_names
        if name not in possible_input_shardings
    ]
    if unsupported_axes:
        raise ValueError(
            "Unsupported mesh axis for AutoParallel Llama3: "
            f"{unsupported_axes}. Supported axes: "
            f"{tuple(possible_input_shardings.keys())}"
        )
    x_sharding = tuple(
        possible_input_shardings[name] for name in dense_mesh.mesh_dim_names
    )

    output_sharding = tuple(
        Shard(1) if name == "tp" else Shard(0) for name in dense_mesh.mesh_dim_names
    )

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        assert traced_inputs is not None
        flat_inputs, _ = torch.utils._pytree.tree_flatten(
            (traced_inputs.args, traced_inputs.kwargs)
        )
        consumed_inputs = {
            desc.idx for desc in get_plain_input_and_grad_nodes(autop.gm.graph)
        }
        replicated_metadata = tuple(Replicate() for _ in range(dense_mesh.ndim))
        input_constraints = []
        for index, value in enumerate(flat_inputs):
            if index not in consumed_inputs:
                input_constraints.append(UNCONSTRAINED)
            elif not isinstance(value, torch.Tensor):
                input_constraints.append(None)
            else:
                input_constraints.append(
                    x_sharding if index < 3 else replicated_metadata
                )
        autop.add_input_constraints(input_constraints)
        autop.add_output_constraints([output_sharding])

        t0 = time.time()
        sharding_placement = autop.optimize_placement(verbose=False)
        t1 = time.time()
        logger.info(f"AutoParallelGraph took {t1 - t0:.2f} seconds")

        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
        )

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism_context=parallelism_context,
    )
    return model
