# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpoint, etc.

import logging

import torch
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
)
from torchtrain.config_manager import JobConfig

from torchtrain.logging_utils import rank0_log
from torchtrain.models.llama.model import Transformer, TransformerBlock

# from pippy.PipelineSchedule import PipelineStageV2Impl
from .pippy_copy import PipelineStageV2Impl

logger = logging.getLogger(__name__)


def distribute_rmsnorm(module, device_mesh):
    # temp sharding API until PTD API is added
    def prepare_input_fn(inputs, device_mesh):
        if isinstance(inputs[0], DTensor):
            return inputs
        elif isinstance(inputs[0], torch.Tensor):
            shard_tensor = DTensor.from_local(
                inputs[0], device_mesh, [Shard(0)], run_check=False
            )
            return shard_tensor
        else:
            raise NotImplementedError("!!")

    def partition_fn(name, module, device_mesh):
        for name, param in module.named_parameters():
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Replicate()])
            )
            module.register_parameter(name, dist_param)

    return distribute_module(
        module,
        device_mesh,
        partition_fn,
        input_fn=prepare_input_fn,
    )


# Uses PTD FSDP AC wrapper
# TODO: why is config needed here?
def checkpoint_wrapper(module, job_config: JobConfig):
    return ptd_checkpoint_wrapper(
        module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False
    )


from typing import List

# this tells which parts of the main model to extract per rank,
# and what input shapes that sub-model expects.
# we can automate this in various ways.  a manual way is also powerful.
pp_config_toy_2_stage = [
    dict(has_embeddings=True, layer_ids=[], has_output=False),
    dict(has_embeddings=False, layer_ids=[0], has_output=True),
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

    def forward(self, input):
        if self.embeddings is not None:
            (tokens,) = input
            h, freqs_cis = self.embeddings(tokens)
        else:
            assert len(input) == 1, input
            (h,) = input
            bsz_seqlen, _ = h.shape
            bsz = 4
            seqlen = bsz_seqlen // bsz
            # TODO how should i get seqlen now
            freqs_cis = self.freqs_cis[:seqlen]

        output = (h,)

        for i in self.layer_ids:
            h = self.layers[i](h, freqs_cis)

        output = (h,)

        if self.norm is not None:
            h = self.norm(h)
            output = self.output(h).float()

        return output


def build_pipeline_stage(world_mesh, model: torch.nn.Module) -> PipelineStageV2Impl:
    pp_stage_meta_inputs = [
        # data batch
        [torch.empty((4, 2048), dtype=torch.int64, device="cuda")],
        # h
        [torch.empty((4 * 2048, 256), dtype=torch.bfloat16, device="cuda")],
    ]

    from .pippy_copy import PipelineStageV2Impl

    # pp_mesh = world_mesh["pp"] if world_mesh.ndim > 1 else world_mesh
    # assert (
    #     pp_mesh.mesh_dim_names[0] == "pp" and pp_mesh.ndim == 1
    # ), pp_mesh.mesh_dim_names
    pp_mesh = world_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()

    # no virtual stages yet
    pp_stage_id = pp_rank
    num_stages = pp_mesh.size()
    pipeline_stage = PipelineStageV2Impl(
        module=model,
        stage_id=pp_stage_id,
        num_stages=num_stages,
        # we need to be clearer about whether rank/world refers to global or local
        rank=world_mesh.get_rank(),
        world_size=world_mesh.size(),
        # annoying
        inputs_meta=pp_stage_meta_inputs[pp_rank],
    )
    return pipeline_stage


def parallelize_llama(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply parallelisms to the model, including PTD parallelisms, and AC.

    NOTE: the model passed in preferrablably shoule be a meta device model,
    otherwise the model needs to be small enough on GPU or can fit into CPU.
    # TODO: apply SP
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
        # First we apply Sequence Parallelism if it's enabled
        tp_mesh = world_mesh["sp"] if world_mesh.ndim > 1 else world_mesh
        sp_degree = job_config.training.sequence_parallelism_degree
        # First:
        # 1. parallelize the first embedding and the last linear proj layer
        # 2. shard the first layer of transformer block
        # TODO: enable loss parallel once it's ready
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embeddings.tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "output": ColwiseParallel(
                    input_layouts=Shard(0),
                    output_layouts=Replicate(),
                ),
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
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(0),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(0)),
                "feed_forward.w3": ColwiseParallel(),
            }

            # adjust num_heads in attention layer to local heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // sp_degree
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // sp_degree

            # shard RMSNorm layers
            distribute_rmsnorm(transformer_block.attention_norm, tp_mesh)
            distribute_rmsnorm(transformer_block.ffn_norm, tp_mesh)

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        rank0_log("Applied Sequence Parallelism to the model...")

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
        assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names
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
        }
        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            for layer_id, maybe_transformer_block in enumerate(model.layers):
                if isinstance(maybe_transformer_block, TransformerBlock):
                    # apply AC to each layer
                    # before wrapping with FSDP, we need to make sure the layer is on GPU
                    transformer_block = maybe_transformer_block.to(device)
                    transformer_block = checkpoint_wrapper(transformer_block, job_config)

                    # Wraps each layer with FSDP
                    model.layers[layer_id] = wrap(transformer_block)

            # wrap the rest layers with FSDP
            model = wrap(model.to(device))
        rank0_log("Applied FSDP to the model...")

    # redundant if FSDP is enabled, but ensure the model is on device regardless of which parallelisms were used
    model.cuda()
    return model
