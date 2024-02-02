# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.

from collections import defaultdict

import torch

# from pippy.PipelineSchedule import (
# get_stage_shapes,
# PipelineStageV2Impl,
# validate_stage_shapes,
# )

from torch.distributed._tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint

from torchtrain.config_manager import JobConfig
from torchtrain.logging_utils import logger
from torchtrain.meta_init import meta_to_real_init_fn

from torchtrain.models.llama.model import Transformer, TransformerBlock

from .pippy_copy import get_stage_shapes, PipelineStageV2Impl, validate_stage_shapes



# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.c10d_functional.reduce_scatter_tensor.default,
}

# Uses PTD FSDP AC wrapper
# currently selective per op and per layer checkpointing are supported
def checkpoint_wrapper(module, config):
    if config.mode == "selective" and config.selective_ac_option == "op":

        def _get_custom_policy(meta):
            def _custom_policy(mode, func, *args, **kwargs):
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return _pt2_selective_checkpoint_context_fn_gen(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            context_fn=selective_checkpointing_context_fn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif config.mode == "full":
        # full AC
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    elif config.mode == "selective" and config.selective_ac_option.isdigit():

        """enables selective checkpointing of candidate layers.
        Usage:
        'selective_ac_option' with a positive 'int' value in config controls which layers to checkpoint.
        1 == checkpointing every one (all).
        2 == checkpoint every 2nd one
        """
        every_x_layer = int(config.selective_ac_option)
        assert (
            every_x_layer >= 0
        ), f"selective layer AC policy (every_x_layer) expects a positive integer, received {every_x_layer}"

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not every_x_layer or checkpoint_wrapper._count % every_x_layer == 0:
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # skip activation checkpointing and store activations for this layer
        else:
            return module

    else:
        raise NotImplementedError(
            "Unknown AC type or AC config. Only selective op and selective layer ac implemented currently."
        )


from typing import List

# this tells which parts of the main model to extract per rank,
# and what input shapes that sub-model expects.
# we can automate this in various ways.  a manual way is also powerful.
pp_config_toy_2_stage = [
    dict(has_embeddings=True, layer_ids=[0], has_output=False),
    dict(has_embeddings=False, layer_ids=[1], has_output=True),
]


class DummyModule(torch.nn.Module):
    def forward(self, *args):
        raise RuntimeError("DummyModule should never be called")


class TransformerChunk(torch.nn.Module):
    def __init__(
        self,
        orig_model: Transformer,
        has_embeddings: bool,
        layer_ids: List[int],
        has_output: bool,
    ):
        super().__init__()

        self.embeddings = orig_model.embeddings if has_embeddings else None

        # this is a constant tensor that's used by all layers, instead of passing it across PP stages
        # just start with one copy per stage
        # TODO hack: model doesn't move this to cuda so we have to handle it ...
        self.freqs_cis = orig_model.embeddings.freqs_cis.cuda()

        # preserve FQNs of original model by preserving structure
        # (including preserving position in layers[] list)- use dummy module
        self.layers = orig_model.layers
        for i in range(len(self.layers)):
            if i not in layer_ids:
                self.layers[i] = DummyModule()

        self.layer_ids = layer_ids
        self.norm = orig_model.norm if has_output else None
        self.output = orig_model.output if has_output else None
        self.dim = orig_model.model_args.dim

    def forward(self, input):
        assert isinstance(input, torch.Tensor), type(input)
        if self.embeddings is not None:
            tokens = input
            h, _ = self.embeddings(tokens)
        else:
            h = input

        assert len(h.shape) == 3, h.shape
        bsz, seqlen, _ = h.shape
        freqs_cis = self.freqs_cis[:seqlen]
        output = h

        # fold batch and sequence dimension for more efficient allgather/reduce_scatter
        folded_h = h.view(-1, self.dim)
        for i in self.layer_ids:
            folded_h = self.layers[i](folded_h, freqs_cis)

        # always output/communicate unfolded shape to preserve shape info for next pp stage
        h = folded_h.view(bsz, seqlen, self.dim)
        output = h

        if self.norm is not None:
            h = self.norm(h)
            output = self.output(h).float()

        return output


def build_pipeline_stage(
    world_mesh, model: torch.nn.Module, loss_fn
) -> PipelineStageV2Impl:
    """
    Notes on get_stage_shapes helper
    - unclear which device microbatch should be on
    - rank and device mapping should be more clear (is it local rank or global rank?)
    - its good to have manual helpers if needed, but user should be able to skip most of this,
      just have it happen on first forward()
    - get_stage_shapes is breaking with my transformer, the way it expects its inputs structured.
      cant we be more flexible?
    - should get_stage_shapes just do validation automatically? is it too expensive? make it a flag?
    i realize that might be hard with the current code structure.
    - if we make you call get_stage_shape manually, we should accept its data repr as input to Stage without manual work
    """
    input_shape = label_shape = (4, 2048)
    microbatch = torch.empty(input_shape, dtype=torch.int64, device="cuda")
    pp_mesh = world_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    device = "cuda"
    # no virtual stages yet
    pp_stage_id = pp_rank
    num_stages = pp_mesh.size()

    shape_meta = get_stage_shapes(
        models=[model],
        stage_ids=[pp_stage_id],
        num_stages=num_stages,
        rank=pp_mesh.get_rank(),
        world_size=pp_mesh.size(),
        group=pp_mesh.get_group(),
        device=device,
        microbatch=[microbatch],
    )
    input_args = [
        torch.empty(s, device=device) for s in shape_meta[pp_stage_id]["inputs"]
    ]
    output_args = [
        torch.empty(s, device=device) for s in shape_meta[pp_stage_id]["outputs"]
    ]
    label_arg = torch.empty(label_shape, device=device, dtype=torch.int64)

    pipeline_stage = PipelineStageV2Impl(
        module=model,
        stage_id=pp_stage_id,
        num_stages=num_stages,
        # we need to be clearer about whether rank/world refers to global or local
        rank=pp_mesh.get_rank(),
        world_size=pp_mesh.size(),
        group=pp_mesh.get_group(),
        device=device,
        input_args=input_args,
        output_args=output_args,
        label_arg=label_arg,
        loss_fn=loss_fn,
    )
    """
    - ValueError: Number of stages (2) must be a multiple of the world_size (4
      - this is thrown when using 2-D parallelism, WS=4 and 2 stages exist.
       - I think we should change to using PP mesh not world ranks

       ok i think i worked around the above by using pg.all_gather,

       but now i am seeing validate_stage_shapes trip up labels send/recv
    """
    # validate_stage_shapes([pipeline_stage], pp_mesh.get_group())

    return pipeline_stage


def parallelize_llama(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply parallelisms to the model, including PTD parallelisms, and AC.

    NOTE: the model passed in preferrablably shoule be a meta device model,
    otherwise the model needs to be small enough on GPU or can fit into CPU.
    """

    # TODO - nice utility for mapping from mesh+parallelisms to local cuda device idx
    # hack: works okish for <8gpu runs for now
    device = "cuda"
    # apply PTD parallelisms
    if parallel_dims.pp_enabled:
        """
        Currently, we just select part of the model for this rank and then continue to wrap that model with SP/DP.

        Later, we have to further wrap it with a PipelineStage object and hook that up to a PipelineSchedule.
        The PipelineSchedule necessarily replaces parts of the main train loop, so it seems reasonable to construct it
        in the trainer file just before using it.

        The PipelineStage object is less clear- we could construct it here, and then make sure it 'acts like a module'
        from the perspective of the TP/DP wrapping, or we can argue that it does not need to mimic a module at all,
        and we just  apply PipelineStages at the point we use the PipelineSchedule.  Then we can work directly with
        more 'vanilla' modules up till that point.

        It should not matter if the model is on meta device or real device at this point- PP stage spliting should
        just leave that alone and let device movement and initialization happen after parallelisms are applied.

        FQNs are important- we aim to preserve original model FQNs so checkpointing is easy.  For virtual stages, we
        add a burden to the trainer code, since there is a list of model chunks rather than a single model chunk. This
        is easy to solve with a helper to apply checkpoint loading/saving to this list.  The FQNs should still match
        the original model if possible, or simply add a PP prefix if needed.

        TODO: however, we still have to deal with one model-chunk per virtual pp stage here in this function. For
        example: world_size=4, PP=2, DP=2, virtual_pp_stages=2 -> we would split the model into 4 PP chunks, putting
        chunks 0, 2 on global ranks 0, 1 and putting chunks 1, 3 on global ranks 2,3.  (Assuming  DP operates between
        global ranks 0-1 and 2-3)

        Note: we use manual model splitting here. It's quite a hack at the moment.  But it could either be cleaned up
        or replaced by pippy's frontend.  In either case, we can try to produce model chunks for stage(s) in this
        function.
        """
        logger.info("Splitting model for pipeline parallelism")
        # todo rebase pytorch and delete slice hack
        pp_mesh = world_mesh["pp"] if world_mesh.ndim > 1 else world_mesh
        assert (
            pp_mesh.mesh_dim_names[0] == "pp" and pp_mesh.ndim == 1
        ), pp_mesh.mesh_dim_names
        pp_rank = pp_mesh.get_local_rank()
        pp_config = pp_config_toy_2_stage
        model = TransformerChunk(model, **pp_config[pp_rank])
        logger.info("Split Model: %s", model)
    if parallel_dims.sp_enabled:
        tp_mesh = world_mesh["sp"]
        sp_degree = job_config.training.sequence_parallel_degree

        # First:
        # 1. parallelize the first embedding and the last linear proj layer
        # 2. shard the first layer of transformer block
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embeddings.tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "output": ColwiseParallel(
                    input_layouts=Shard(0),
                    output_layouts=Shard(-1)
                    if parallel_dims.loss_parallel_enabled
                    else Replicate(),
                    use_local_output=not parallel_dims.loss_parallel_enabled,
                ),
                "norm": SequenceParallel(sequence_dim=0),
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(), None),
                    desired_input_layouts=(Shard(0), None),
                    use_local_output=True,
                ),
            },
        )

        # apply sequence parallelism to every transformer block
        for layer_id, transformer_block in enumerate(model.layers):
            layer_plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(0), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(0)),
                "attention_norm": SequenceParallel(sequence_dim=0),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(0),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(0)),
                "feed_forward.w3": ColwiseParallel(),
                "ffn_norm": SequenceParallel(sequence_dim=0),
            }

            # adjust num_heads in attention layer to local heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // sp_degree
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // sp_degree

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        logger.info("Applied Sequence Parallelism to the model")

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]

        fsdp_config = {
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16,
                # TODO: see whether we should expose a option to user
                reduce_dtype=torch.float32,
            ),
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            # When torch.compile is active, it requires us to set use_orig_params=True
            "use_orig_params": True,
            "device_mesh": dp_mesh,
            "param_init_fn": meta_to_real_init_fn,
        }
        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            for layer_id, maybe_transformer_block in enumerate(model.layers):
                if isinstance(maybe_transformer_block, TransformerBlock):
                    # apply AC to each layer
                    # before wrapping with FSDP, we need to make sure the layer is on GPU
                    transformer_block = maybe_transformer_block.to(device)
                    if job_config.activation_checkpoint.mode in ("full", "selective"):
                        # wrap the transformer block with checkpoint wrapper, using config settings
                        transformer_block = checkpoint_wrapper(
                            transformer_block, job_config.activation_checkpoint
                        )

                    # Wraps each layer with FSDP
                    model.layers[layer_id] = wrap(transformer_block)

            # wrap the rest layers with FSDP
            model = wrap(model.to(device))
        logger.info("Applied FSDP to the model")


    else:
        meta_to_real_init_fn(model)
        model.cuda()

    # we have now moved from meta to device,
    # reset parameters for proper initialization
    model.reset_parameters()
    logger.info("Model fully initialized via reset_parameters")

    return model
