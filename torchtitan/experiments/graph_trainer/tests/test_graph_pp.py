# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from contextlib import nullcontext
from functools import partial
from typing import Callable, Optional

import autoparallel._testing.models.dsv3 as dsv3_module
import torch
import torch.distributed._tools.fake_collectives
import torch.nn as nn
from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    DeepSeekV3Stage0,
    DeepSeekV3StageI,
    DeepSeekV3StageN,
    dsv3_loss_fn,
    MoEArgs,
)
from autoparallel.api import move_to_fake
from autoparallel.api_pp import AutoParallelPP, make_pp_module
from autoparallel.graph_passes.graph_pp_runner import (
    get_multiplexed_graph_callables,
    GraphCallables,
    GraphMeta,
    GraphPipelineStage,
    GraphPPRunner,
    overlap_fw_bw,
    stage_backward_input,
    stage_backward_weight,
    stage_forward,
    stage_full_backward,
    stage_reduce_grad,
    stage_reshard,
    stage_unshard,
)
from autoparallel.shardings.placement_options import NumericsLogger
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    get_schedule_class,
    OVERLAP_F_B,
    PipelineScheduleMulti,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def assign_logical_stages_to_pp_rank(
    schedule_name: str, pp_degree: int, stages_per_rank: int
) -> dict[int, list[int]]:
    style = "v" if schedule_name in ("ZBVZeroBubble", "DualPipeV") else "loop"
    if style == "loop":
        pp_rank_to_stage_indices = {
            pp_rank: [pp_rank + s * pp_degree for s in range(stages_per_rank)]
            for pp_rank in range(pp_degree)
        }
    elif style == "v":
        total_pp_stages = pp_degree * stages_per_rank
        pp_rank_to_stage_indices = {
            pp_rank: [pp_rank, total_pp_stages - 1 - pp_rank]
            for pp_rank in range(pp_degree)
        }
    return pp_rank_to_stage_indices


def build_pipeline_schedule(
    stages: list[PipelineStage],
    loss_fn: Callable,
    pipeline_parallel_schedule: str,
    microbatch_size: int,
    local_batch_size: int,
    pipeline_parallel_degree: int,
    backward_requires_autograd: bool = False,
    scale_grads: bool = True,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given configuration and stages."""
    schedule_class = get_schedule_class(pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    assert looped_schedule, "Only looped schedules are supported"
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if local_batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by {microbatch_size=}. "
        )
    n_microbatches = local_batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all pp ranks
    num_total_stages = pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        backward_requires_autograd=backward_requires_autograd,
        scale_grads=scale_grads,
    )
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )
    return schedule


def run_test(
    fake_evaluate: bool,
    use_loss_fn: bool,
    schedule_name: str,
    rng_seed: Optional[int],
    logs_dir: str,
    use_cache: bool,
    use_inductor: bool = False,
):
    if not fake_evaluate:
        pp_degree = 2
        dp_mod_ep_degree = 2
        ep_degree = 2
    else:
        pp_degree = 4
        dp_mod_ep_degree = 4
        ep_degree = 64

    dp_degree = dp_mod_ep_degree * ep_degree
    world_size = pp_degree * dp_mod_ep_degree * ep_degree

    # Initialize process group based on evaluation mode
    if fake_evaluate:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 4"
        assert (
            int(os.getenv("WORLD_SIZE")) == pp_degree
        ), "world_size must be 4, for fake evaluation"
        rank = int(os.getenv("RANK"))
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake",
            store=fake_store,
            rank=rank * dp_degree,  # global rank is pp_rank * spmd_size
            world_size=world_size,
        )
        pp_rank = rank
    else:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 8"
        assert (
            int(os.getenv("WORLD_SIZE")) == world_size
        ), "Need at least 8 GPUs for real evaluation"
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="nccl")

    # Initialize device mesh (common for both modes)
    world_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (pp_degree, dp_mod_ep_degree, ep_degree),
        mesh_dim_names=(
            "pp",
            "dp_mod_ep",
            "ep",
        ),
    )

    # Set pp_rank based on evaluation mode
    if not fake_evaluate:
        pp_rank = world_mesh["pp"].get_local_rank()

    stages_per_rank = 2
    total_pp_stages = pp_degree * stages_per_rank

    # This is the spmd mesh to be used for tracing
    mesh = world_mesh[("dp_mod_ep", "ep")]

    # Batch size that will be supplied to the schedule and will be broken down into microbatches
    local_batch_size = 32
    # global_batch_size = local_batch_size * dp_degree
    n_microbatches = 16
    # Batch size with which the spmd graphs will actually be executed
    microbatch_size = local_batch_size // n_microbatches
    assert (
        microbatch_size >= 1
    ), f"invalid config {local_batch_size=}, {n_microbatches=}"
    # Batch size to be used for spmd tracing
    spmd_batch_size = microbatch_size * dp_degree

    seq_len = 1024

    if fake_evaluate:
        config = DeepSeekV3ModelArgs(
            vocab_size=102400,
            max_seq_len=seq_len,
            dim=2048,
            inter_dim=10944,
            moe_inter_dim=1408,
            n_layers=8,  # 27,
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
    else:
        config = DeepSeekV3ModelArgs(
            vocab_size=2048,
            max_seq_len=seq_len,
            dim=256,
            inter_dim=1024,
            moe_inter_dim=256,
            n_layers=4,
            n_dense_layers=0,  # 1,
            n_heads=16,
            moe_args=MoEArgs(
                num_experts=4,
                num_shared_experts=2,
                top_k=2,
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
        )

    with torch.device("meta"):
        model = DeepSeekV3Model(config).bfloat16()
        embed, layers, norm, output = list(model.children())
        items = list(layers.items())
        assert len(items) == config.n_layers
        n_layers_per_rank = len(items) // total_pp_stages
        layers = [
            nn.ModuleDict(items[i : i + n_layers_per_rank])
            for i in range(0, len(items), n_layers_per_rank)
        ]
        assert len(layers) == total_pp_stages
        for lst in layers:
            assert len(lst) * len(layers) == config.n_layers

    def make_input_fn(
        batch_size: int,
        inp_type: str,
        device: torch.device,
    ):
        """
        Factory to create input/output generator functions for pipeline stages.

        Args:
            batch_size: Batch size (spmd_batch_size, local_batch_size, or microbatch_size)
            inp_type: One of "tokens", "embeddings", or "logits"
            device: Device to create tensors on (cuda device or "meta")
        """

        def input_fn() -> torch.Tensor:
            if inp_type == "tokens":
                return torch.randint(
                    0,
                    config.vocab_size,
                    (batch_size, seq_len),
                    device=device,
                )
            elif inp_type == "embeddings":
                return torch.randn(
                    (batch_size, seq_len, config.dim),
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
            elif inp_type == "logits":
                return torch.randn(
                    (batch_size, seq_len, config.vocab_size),
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
            elif inp_type == "loss":
                return torch.scalar_tensor(
                    1.0,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )
            else:
                raise ValueError(f"Unknown input type: {inp_type}")

        return input_fn

    # Target generators (if needed for loss computation)
    tracing_target_fn = make_input_fn(spmd_batch_size, "tokens", device)
    runtime_target_fn = make_input_fn(local_batch_size, "tokens", device)

    # Tracing input functions
    tracing_input_fn_fist_stage = make_input_fn(spmd_batch_size, "tokens", device)
    tracing_input_fn_intermediate_stage = make_input_fn(
        spmd_batch_size, "embeddings", device
    )

    def last_stage_inp_with_loss_fn():
        return (
            tracing_input_fn_intermediate_stage(),
            tracing_target_fn(),
        )

    tracing_input_fn_last_stage = (
        last_stage_inp_with_loss_fn
        if use_loss_fn
        else tracing_input_fn_intermediate_stage
    )

    # Runtime input function
    runtime_input_fn_first_stage = make_input_fn(local_batch_size, "tokens", device)

    # Shape inference functions
    meta_device = torch.device("meta")
    shape_inference_input_fn_first_stage = make_input_fn(
        microbatch_size, "tokens", meta_device
    )
    shape_inference_fn_intermediate_stage = make_input_fn(
        microbatch_size, "embeddings", meta_device
    )
    shape_inference_output_fn_last_stage = (
        make_input_fn(0, "loss", meta_device)
        if use_loss_fn
        else make_input_fn(microbatch_size, "logits", meta_device)
    )

    # Step 1. Construct the logical pipeline stages
    with torch.device("meta"):
        virtual_pp_stages = [DeepSeekV3Stage0(embed, layers[0], config)]
        for i in range(1, total_pp_stages - 1):
            virtual_pp_stages.append(DeepSeekV3StageI(layers[i], config))
        last_stage = DeepSeekV3StageN(layers[total_pp_stages - 1], norm, output, config)
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

            last_stage = ModelWithLoss(last_stage)
        virtual_pp_stages.append(last_stage)
    # Step 2. Assign each logical stage(s) to pp ranks for the given schedule
    pp_rank_to_stage_indices = assign_logical_stages_to_pp_rank(
        schedule_name, pp_degree, stages_per_rank
    )
    print(pp_rank_to_stage_indices)
    assert len(pp_rank_to_stage_indices) == pp_degree
    for stages in pp_rank_to_stage_indices.values():
        assert len(stages) * pp_degree == len(virtual_pp_stages)
    stage_indices_current_pp_rank = pp_rank_to_stage_indices[pp_rank]
    should_log_weights = should_log_fw_outs = False
    if rng_seed:
        # Compute the ranks to log from
        # 1. for fw_outs, log from coord [pp_rank_containing_last_stage, 0, 0]
        last_stage_idx = total_pp_stages - 1
        pp_rank_containing_last_stage = None
        for pp_rank_, stage_indices in pp_rank_to_stage_indices.items():
            if last_stage_idx in stage_indices:
                assert pp_rank_containing_last_stage is None
                pp_rank_containing_last_stage = pp_rank_

        log_fw_out_rank_coordinate = []
        for mesh_dim_name in world_mesh.mesh_dim_names:
            if mesh_dim_name == "pp":
                log_fw_out_rank_coordinate.append(pp_rank_containing_last_stage)
            else:
                log_fw_out_rank_coordinate.append(0)
        should_log_fw_outs = world_mesh.get_coordinate() == log_fw_out_rank_coordinate

        # 2. for weights, log from coords [:, 0, 0]
        pp_world_size = world_mesh.shape[world_mesh._get_mesh_dim_by_name("pp")]
        log_weights_rank_coordinates = [(i, 0, 0) for i in range(pp_world_size)]
        should_log_weights = (
            tuple(world_mesh.get_coordinate()) in log_weights_rank_coordinates
        )

    stage_mods: dict[int, torch.nn.Module] = {}
    stage_graphs: dict[int, GraphCallables] = {}
    stage_graph_metas: dict[int, GraphMeta] = {}
    # Step 3. Apply AutoParallel to each logical stage assigned to this pp rank
    root_cache = "tmp"
    os.makedirs(root_cache, exist_ok=True)

    for stage_idx in stage_indices_current_pp_rank:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"begin_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )
        stage_mod = virtual_pp_stages[stage_idx]
        eval_mode = "fake" if fake_evaluate else "real"
        stage_file = os.path.join(root_cache, f"stage_{eval_mode}_{stage_idx}.pth")
        if os.path.exists(stage_file) and use_cache:
            cache = torch.load(stage_file, weights_only=False)
            graph_callables = cache["graph_callables"]
            graph_meta = cache["graph_meta"]
            cache["sharded_param_dict"] = {
                k: nn.Parameter(v.detach())
                for k, v in cache["sharded_param_dict"].items()
            }
            fake_mode = FakeTensorMode()
            stage_mod = move_to_fake(stage_mod, fake_mode, device)
            pp_mod = make_pp_module(
                cache["sharded_param_dict"],
                cache["sharded_buffer_dict"],
                stage_mod,
            )
        else:
            if stage_idx == 0:
                input_fn = tracing_input_fn_fist_stage
            elif stage_idx == total_pp_stages - 1:

                input_fn = tracing_input_fn_last_stage

            else:
                input_fn = tracing_input_fn_intermediate_stage
            with AutoParallelPP(
                stage_mod,
                input_fn,
                mesh,
                dynamic=True,
                reshard_after_forward=False,
            ) as autop:
                autop.add_parameter_memory_constraint(low=None, high=None)

                # x_sharding = (Shard(0), Replicate())
                x_sharding = (Shard(0), Shard(0))
                if use_loss_fn and stage_idx == total_pp_stages - 1:
                    autop.add_input_constraints([x_sharding, x_sharding])
                    autop.add_output_constraints([(Replicate(), Replicate())])
                else:
                    autop.add_input_constraints([x_sharding])
                    autop.add_output_constraints([x_sharding])

                sharding_placement = autop.optimize_placement(verbose=False)
                graph_passes = ["split_fsdp_collectives"]
                if stage_idx > 0:
                    # First stage does not produce gradients wrt to input,
                    # hence we do not do apply the split_dI_dW pass
                    graph_passes.extend(["split_dI_dW"])
                cache = autop.apply_placement_pp(
                    sharding_placement=sharding_placement, graph_passes=graph_passes
                )
                graph_callables = cache["graph_callables"]
                graph_meta = cache["graph_meta"]
                pp_mod = autop.parallel_model
                if use_cache:
                    torch.save(cache, stage_file)

        pp_mod.to_empty(device=device)
        # run weight init on our sharded DTensor params
        pp_mod.init_weights(buffer_device=device, seed=rng_seed)

        # Store each stage's information in stage_mods, stage_graphs, and stage_graph_metas
        stage_mods[stage_idx] = pp_mod
        stage_graphs[stage_idx] = GraphCallables(
            fw=graph_callables["fw"],
            full_bw=graph_callables["full_bw"],
            bw_dI=graph_callables["bw_dI"],
            bw_dW=graph_callables["bw_dW"],
            unshard=graph_callables["unshard"],
            reduce_grad=graph_callables["reduce_grad"],
        )
        stage_graph_metas[stage_idx] = GraphMeta(
            num_mutate_inputs=graph_meta["num_mutate_inputs"],
            num_user_outputs=graph_meta["num_user_outputs"],
            num_symints_saved_for_bw=graph_meta["num_symints_saved_for_bw"],
            num_params=graph_meta["num_params"],
            num_buffers=graph_meta["num_buffers"],
            num_input_grads=graph_meta["num_input_grads"],
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"end_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )

    # Two stages per pp rank
    assert (
        len(stage_indices_current_pp_rank)
        == len(stage_mods)
        == len(stage_graphs)
        == len(stage_graph_metas)
    )

    world_size = torch.distributed.get_world_size()
    num_world_stages = world_size * len(stage_mods)

    numerics_logger = None
    if rng_seed is not None:
        numerics_logger = NumericsLogger(logs_dir)
        numerics_logger.log_pp_model_weights(
            model, stage_mods, num_world_stages, should_log=should_log_weights
        )
        torch.manual_seed(rng_seed)

    stages = []
    # Step 4. Construct pipeline stages for this pp_rank using the stage modules, graphs and metadata
    for pp_stage_idx, pp_stage_mod in stage_mods.items():
        stage = GraphPipelineStage(
            pp_stage_mod,
            stage_graphs[pp_stage_idx],
            stage_graph_metas[pp_stage_idx],
            stage_index=pp_stage_idx,
            num_stages=len(virtual_pp_stages),
            device=device,
            input_args=(
                shape_inference_input_fn_first_stage()
                if pp_stage_idx == 0
                else shape_inference_fn_intermediate_stage()
            ),
            output_args=(
                shape_inference_output_fn_last_stage()
                if pp_stage_idx == (total_pp_stages - 1)
                else shape_inference_fn_intermediate_stage()
            ),
            group=world_mesh.get_group("pp"),
            numerics_logger=numerics_logger,
            should_log_fw_outs=should_log_fw_outs,
        )
        stages.append(stage)

    # Step 5. Construct the pipeline runner using the pipeline stages for this pp_rank
    schedule = build_pipeline_schedule(
        stages=stages,
        loss_fn=None,
        pipeline_parallel_schedule=schedule_name,
        microbatch_size=microbatch_size,
        local_batch_size=local_batch_size,
        pipeline_parallel_degree=pp_degree,
        backward_requires_autograd=False,
        scale_grads=rng_seed is None,  # In determinism mode, don't scale grads
    )
    assert isinstance(schedule, _PipelineScheduleRuntime)

    # Step 6. Override the pipeline runner's action implementations
    schedule.register_custom_function(FORWARD, stage_forward)
    schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
    schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
    schedule.register_custom_function(RESHARD, stage_reshard)
    schedule.register_custom_function(UNSHARD, stage_unshard)
    schedule.register_custom_function(BACKWARD_INPUT, stage_backward_input)
    schedule.register_custom_function(BACKWARD_WEIGHT, stage_backward_weight)
    if schedule_name == "DualPipeV":
        from autoparallel.graph_passes.graph_multiplex import multiplex_fw_bw_graph

        multiplexed_graph_callables = get_multiplexed_graph_callables(
            stage_graphs,
            partial(multiplex_fw_bw_graph, overlap_with_annotations=True),
        )
        schedule.register_custom_function(
            OVERLAP_F_B, partial(overlap_fw_bw, multiplexed_graph_callables)
        )

    # Step 7. Register the schedule with the graph runner
    graph_pp_runner = GraphPPRunner(schedule, inductor=use_inductor)

    # Step 8. Run the whole pipeline once using the graph runner
    has_last_stage = (total_pp_stages - 1) in stage_mods
    execution_fake_mode = (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    )

    with execution_fake_mode:
        with torch.no_grad():
            target, losses = (
                (runtime_target_fn(), [])
                if has_last_stage and use_loss_fn
                else (None, None)
            )
            if pp_rank == 0:
                x = runtime_input_fn_first_stage()
                if numerics_logger is not None:
                    numerics_logger.log_diff(
                        x.to(torch.float32), prefix="full batch input"
                    )
                graph_pp_runner.step(
                    x, target=target, losses=losses, return_outputs=False
                )
            else:
                graph_pp_runner.step(target=target, losses=losses, return_outputs=False)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "pipeline_step_losses",
                    "encoding": "string",
                },
                payload_fn=lambda: f"losses: {losses}",
            )
        if numerics_logger is not None:
            numerics_logger.log_pp_grads(
                model, stage_mods, num_world_stages, should_log=should_log_weights
            )

    print("All good!")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek V3 pipeline parallel example"
    )
    parser.add_argument(
        "--fake-evaluate",
        action="store_true",
        default=False,
        help="Use fake evaluation mode with FakeTensorMode (default: False)",
    )
    parser.add_argument(
        "--use-loss-fn",
        action="store_true",
        default=False,
        help="Trace loss_fn as part of model forward graph for the last stage (default: False)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Use a specific rng seed and deterministic algorithms for run-to-run invariance (default: None).",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="out/",
        help="Directory to store logs (default: ./out/).",
    )
    parser.add_argument(
        "--schedule-name",
        type=str,
        default="DualPipeV",
        choices=["Interleaved1F1B", "ZBVZeroBubble", "DualPipeV"],
        help="Schedule to use for PP",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Use cached graph files if available (default: False)",
    )
    parser.add_argument(
        "--inductor",
        action="store_true",
        default=False,
        help="Compile subgraphs with Inductor (also forces balanced MoE routing)",
    )
    args = parser.parse_args()

    if args.use_cache and not args.fake_evaluate:
        parser.error("--use-cache can only be used with --fake-evaluate")

    if args.rng_seed is not None:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.rng_seed)

    if args.inductor:
        # The DSv3 MoE implementation uses .tolist() and data-dependent grouped_mm
        # offsets, which Inductor cannot compile. Force balanced routing to make
        # all token counts static.
        dsv3_module.FORCE_BALANCED_ROUTING = True

    run_test(
        fake_evaluate=args.fake_evaluate,
        use_loss_fn=args.use_loss_fn,
        schedule_name=args.schedule_name,
        rng_seed=args.rng_seed,
        logs_dir=args.logs_dir,
        use_cache=args.use_cache,
        use_inductor=args.inductor,
    )
