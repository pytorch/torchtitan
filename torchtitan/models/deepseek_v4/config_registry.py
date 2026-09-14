# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry
from .mtp import MTPLoss


def deepseek_v4_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_mtp_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len, n_mtp_layers=1)
    return Trainer.Config(
        loss=MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_flash(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("deepseek_v4_flash", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_pro(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("deepseek_v4_pro", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


_GB300_FLEX_KERNEL_OPTIONS = {
    # Forward tile.
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_stages": 1,
    "num_warps": 4,
    # Backward tiles. BLOCK_M/BLOCK_N govern only the forward kernel, so
    # without these the backward autotunes freely and hard-crashes the GPU
    # with "CUDA error: unspecified launch failure" at head_dim=512.
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}


def _pin_gb300_flex_tiles(config: Trainer.Config) -> None:
    """Pin the FlexAttention Triton tiles on every flex layer of `config`.

    Both `pro` and `flash` have head_dim=512, and on GB300 (232448 B of shared
    memory per block) Inductor finds no valid Triton config for the default
    autotune sweep -- the first forward dies with "No valid triton configs.
    OutOfMemoryError: out of resource: triton_flex_attention Required: 294912
    Hardware limit: 232448".

    Measured on one GB300 at D=512 with a block mask: every larger forward tile
    fails (64x64, 64x32 and 128x32 all raise NoValidChoicesError or a launch
    failure, with or without num_stages=1), and 32x32 fails unless num_stages
    and num_warps are pinned too.

    The backward needs its own tiles. Measured at `pro`'s real shape (H=128,
    D=512, Q=4096, KV=5121 -- KV is longer than Q because of index_topk=1024
    and the compressed paths), the forward compiles fine with the tile above
    while the backward dies with "CUDA error: unspecified launch failure";
    pinning BLOCK_M1/N1/M2/N2 fixes it. torchtitan's FlexAttention docstring
    names this the intended workflow: autotune once, then set kernel_options
    explicitly.

    This is a correctness requirement, not a tuning choice -- without it the
    model cannot execute a forward pass on this hardware. It is also small
    enough to cost attention throughput, so revisit if a future PyTorch lowers
    the shared-memory demand at head_dim=512.
    """
    for layer in config.model_spec.model.layers:
        attention = getattr(layer, "attention", None)
        inner = getattr(attention, "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.kernel_options = dict(_GB300_FLEX_KERNEL_OPTIONS)



def deepseek_v4_pro_64xgb300(seq_len: int | None = None) -> Trainer.Config:
    """`deepseek_v4_pro` made to fit on 64x GB300 (16 nodes x 4 GPUs).

    Derived from the stock ``deepseek_v4_pro`` so the delta stays auditable;
    the model itself is untouched. Three changes, each forced by a measured
    constraint rather than by taste:

    1. ``training.dtype = "bfloat16"``. This is the change that makes the model
       fit at all, and no amount of parallelism substitutes for it. Sharding
       divides the model state, it does not shrink it: `pro` is 1.573 T
       parameters, and torchtitan's default ``training.dtype="float32"`` costs
       16 B/param (fp32 shard + fp32 grad + two fp32 AdamW moments) = 25.17 TB.
       Spread perfectly over all 64 GPUs that is 393 GB/GPU against 298 GB of
       HBM -- 132 % of the machine before a single activation. Full bf16 is
       8 B/param = 12.58 TB = 196.6 GB/GPU, leaving ~101 GB/GPU of headroom.
       ``mixed_precision_param`` is already bfloat16, so this only drops the
       extra fp32 master copy, exactly as documented on the field.

    2. ``expert_parallel_degree = 64``. Stock is 1, which leaves all 384
       experts of a layer to FSDP: one all-gather would materialize
       384 x 3 x 7168 x 3072 x 2 B = 50.7 GB for a single layer, more with
       prefetch, which does not survive a 101 GB budget. At EP=64 each rank
       owns 384/64 = 6 whole experts, so the expert stack is never all-gathered
       and only token dispatch crosses the wire. EP must divide
       ``dp_shard * cp * tp`` (= 64 here), and 64 | 384, so the mesh is legal.

    3. ``activation_checkpoint = FullAC``. Stock is ``None``. 61 blocks of
       dim 7168 with 128 heads x 512 head_dim cannot keep their forward
       activations in what is left after the weights. FullAC (rather than the
       SelectiveAC that upstream's ``deepseek_v3_671b`` uses) because the
       headroom here is far tighter than that recipe assumes.

    ``disable_cuda_graphs`` follows upstream's own ``deepseek_v3_671b`` recipe.
    Everything else -- optimizer, LR schedule, loss, dataloader, compile
    settings, batch shape -- is inherited from stock ``deepseek_v4_pro``.

    Note: bf16 AdamW moments are fine for a throughput baseline but are not a
    convergence-grade choice for a real 1.573 T pretrain.
    """
    config = deepseek_v4_pro(seq_len=seq_len)
    config.training.dtype = "bfloat16"
    config.training.disable_cuda_graphs = True
    config.parallelism = ParallelismConfig(expert_parallel_degree=64)
    config.activation_checkpoint = FullAC.Config()

    _pin_gb300_flex_tiles(config)
    return config


def deepseek_v4_flash_64xgb300(seq_len: int | None = None) -> Trainer.Config:
    """`deepseek_v4_flash` (284.3 B) on 64x GB300, same recipe as the Pro baseline.

    Companion to `deepseek_v4_pro_64xgb300`: same four deltas off stock, so the
    two flavors differ only in the model and a TFLOP/s comparison between them
    is honest. The numbers behind each delta are different, though, and only
    one of the four is actually forced at this size:

    1. ``training.dtype = "bfloat16"``. For `pro` this was the change that made
       the model fit at all. Here it is NOT forced: 284.3 B params at
       torchtitan's default 16 B/param (fp32 shard + fp32 grad + two fp32 AdamW
       moments) is 4.55 TB = 71.1 GB/GPU over 64 GPUs, which sits inside 298 GB
       of HBM with room to spare -- where `pro` needed 393 GB/GPU and did not.
       It is kept anyway so the comparison against the 31.48 TFLOP/s Pro
       baseline measures the model and not the optimizer dtype. Full bf16 is
       8 B/param = 2.27 TB = 35.5 GB/GPU. To spend the headroom on convergence
       instead of comparability, pass ``--training.dtype float32``.

    2. ``expert_parallel_degree = 64``. Stock is 1, which leaves all 256 routed
       experts of a layer to FSDP: one all-gather materializes
       256 x 3 x 4096 x 2048 x 2 B = 12.9 GB for a single layer, more with
       prefetch. At EP=64 each rank owns 256/64 = 4 whole experts, 0.20 GB, and
       only token dispatch crosses the wire. EP must divide
       ``dp_shard * cp * tp`` (= 64 here), and 64 | 256, so the mesh is legal.
       Less load-bearing than at `pro`'s 50.7 GB/layer, but the same argument.

    3. ``activation_checkpoint = FullAC``. Stock is ``None``. `pro` measured
       ~56.9 GiB of activations WITH FullAC; `flash` has 43 blocks instead of
       61 at dim 4096 instead of 7168, so expect roughly 0.4x that. That is far
       inside the ~260 GB/GPU this flavor leaves free, so unlike `pro` this is a
       safety choice rather than a necessity -- dropping AC is the first thing
       to try for throughput, and the batch shape (stock: 1 x 4096 tokens per
       rank) is the second, via
       ``--training.num_tokens_per_microbatch_per_dp_rank``.

    4. The pinned FlexAttention tiles, via ``_pin_gb300_flex_tiles``. This one
       IS forced: `flash` has ``head_dim=512`` exactly like `pro`, and on GB300
       Inductor finds no valid Triton config at that head_dim for the default
       autotune sweep, so the first forward dies before it can OOM. See
       ``_GB300_FLEX_KERNEL_OPTIONS`` for the measured sweep behind the tiles.

    ``disable_cuda_graphs`` follows upstream's own ``deepseek_v3_671b`` recipe.
    Everything else -- optimizer, LR schedule, loss, dataloader, compile
    settings, batch shape -- is inherited from stock ``deepseek_v4_flash``.

    Note: bf16 AdamW moments are fine for a throughput baseline but are not a
    convergence-grade choice for a real 284 B pretrain.
    """
    config = deepseek_v4_flash(seq_len=seq_len)
    config.training.dtype = "bfloat16"
    config.training.disable_cuda_graphs = True
    config.parallelism = ParallelismConfig(expert_parallel_degree=64)
    config.activation_checkpoint = FullAC.Config()
    _pin_gb300_flex_tiles(config)
    return config


# ===========================================================================
# deepseek_v4_flash (284.3 B) at seq_len 8192 on 64x GB300 -- optimization round 1.
#
# The 4096 baseline measured **23.74 TFLOP/s** at **56.14 GiB (20.3 %)** peak
# with 0 allocator retries (job 305). That memory figure is the headline: this
# flavor uses a fifth of the GPU, where `pro` sat at 84 %. So the levers that
# were unaffordable on `pro` are free here, and the ones that mattered there
# are worth re-testing at a completely different operating point.
#
# Each config below changes exactly one thing off the 8k baseline so the deltas
# are attributable, plus one combined run.
#
# Carried over as priors, not assumptions:
#   * EP=16 was the optimum for BOTH `deepseek_v4_pro` and Kimi K2.7 1T on this
#     cluster (+13.7 % and +8.8 %). 4 GPUs/node, no NVSwitch, 4 IB HCAs -- EP
#     width sets how far each MoE dispatch travels. 256 experts / 16 = 16 per
#     rank, and EP must divide dp_shard*cp*tp = 64, so it is legal.
#   * `block_size` 128 -> 32 was worth **+19 %** on `pro` (37.95 -> 45.30),
#     by making the DSA sparse mask fine enough to skip score area.
#   * Dropping FullAC removes a whole extra forward pass. On `pro` this OOMed
#     by >91 GiB; here there is ~220 GiB free, so it should simply work. On
#     Kimi the equivalent move (freeing memory, then spending it) was the
#     difference between 179 and 305 TFLOP/s.
#   * bf16 gradient reduction: +0.6 % on `pro`, +2.9-4.7 % on Kimi. It pays in
#     proportion to how much of the step is comms.
#
# Autotune is deliberately left ON, matching the baseline. Flash does not pin
# its flex tiles and its autotuner picks BLOCK_M=128/BLOCK_N=64 -- far larger
# than the 32x32 `pro` is limited to at n_heads=128 -- so pinning `pro`'s tiles
# here would likely be a regression. The cost is a ~17 % run-to-run spread and
# several minutes of startup; pinning flash-appropriate tiles is its own
# experiment, not a freebie.
# ===========================================================================


def _flash_8k(seq_len: int | None = 8192) -> Trainer.Config:
    """The flash 64xGB300 recipe at seq_len 8192."""
    return deepseek_v4_flash_64xgb300(seq_len=seq_len)


def deepseek_v4_flash_8k(seq_len: int | None = 8192) -> Trainer.Config:
    """F0. The 8k REFERENCE: baseline recipe, seq_len 8192.

    Everything else in this round is measured against this, not against the
    4096 baseline's 23.74 TFLOP/s -- doubling the context changes the attention
    work per token, so a cross-length comparison would not be clean.

    Note `index_topk=512` against `seq_len // 4` = 2048, so DSA's top-k is a
    genuine selection at 8192 just as it was at 4096 (512 of 1024). Unlike
    `pro`, which had `index_topk=1024` and therefore an inert top-k at 4096,
    flash does not change sparsity regime between the two lengths.
    """
    return _flash_8k(seq_len)


def deepseek_v4_flash_8k_no_ac(seq_len: int | None = 8192) -> Trainer.Config:
    """F1. Drop FullAC. The largest lever available, given 20 % memory use.

    FullAC recomputes the whole forward: on `pro`'s profile that showed up as
    244 attention forward calls against 122 backward. Removing it should cut
    close to a forward pass of work. It was impossible on `pro` (OOM by
    >91 GiB) and unaffordable on Kimi until bf16 masters freed 92 GiB.
    """
    config = _flash_8k(seq_len)
    config.activation_checkpoint = None
    return config


def deepseek_v4_flash_8k_ep16(seq_len: int | None = 8192) -> Trainer.Config:
    """F2. EP 64 -> 16, the optimum found for both other MoE models here."""
    config = _flash_8k(seq_len)
    config.parallelism = ParallelismConfig(expert_parallel_degree=16)
    return config


def deepseek_v4_flash_8k_blocksize32(seq_len: int | None = 8192) -> Trainer.Config:
    """F3. FlexAttention `block_size` 128 -> 32, worth +19 % on `pro`.

    Finer sparse-mask granularity means the kernel visits less score area. On
    `pro` the visited/selected ratio went 1.72x -> 1.20x. Flash shares
    `head_dim=512` and the same DSA structure, so the mechanism should carry --
    but flash autotunes larger tiles, and a BlockMask block cannot be smaller
    than the compute tile, so this may force smaller tiles as a side effect.
    That interaction is exactly what the run measures.
    """
    config = _flash_8k(seq_len)
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.block_size = 32
    return config


def deepseek_v4_flash_8k_bf16reduce(seq_len: int | None = 8192) -> Trainer.Config:
    """F4. `mixed_precision_reduce="bfloat16"`, halving gradient all-reduce bytes.

    Upstream types the field `Literal["float32"]`; widening it in
    `torchtitan/config/configs.py` is the whole change. **Changes training
    numerics** -- gradients reduce-scattered across 64 shards in bf16 accumulate
    rounding error that fp32 reduction exists to avoid. Fine for a throughput
    measurement; needs a fixed-seed fp32 control before a real run.
    """
    config = _flash_8k(seq_len)
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def deepseek_v4_flash_8k_all(seq_len: int | None = 8192) -> Trainer.Config:
    """F5. Everything at once: no AC + EP=16 + block_size 32 + bf16 reduce.

    Run alongside the single-variable configs rather than instead of them: if
    this beats them all the components compose, and if it does not the
    single-variable runs say which one is fighting the others. On Kimi the two
    comms levers turned out to overlap (bf16 reduce was +4.7 % alone but only
    +2.9 % on top of a larger batch), so composition is not a given.
    """
    config = _flash_8k(seq_len)
    config.activation_checkpoint = None
    config.parallelism = ParallelismConfig(expert_parallel_degree=16)
    config.training.mixed_precision_reduce = "bfloat16"
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.block_size = 32
    return config


# --- round 1b: what the no-AC failure and the memory headroom point at -------
#
# F1 (no AC) OOMed at step 1 with the GPU at 275.26 of 276.50 GiB. Model memory
# is 10.66 GiB, so 43 layers at 8192 tokens want **~265 GiB** of activations
# without checkpointing, against FullAC's ~84 GiB -- roughly 3x, needing
# ~180 GiB more than exists. FullAC is load-bearing at 8k, so the lever is the
# middle rung rather than removing it.
#
# The 8k reference sits at 95.14 GiB (34.4 %), leaving ~180 GiB. Batch is the
# most reliable lever found across all three models on this cluster (+58 %
# cumulative on Kimi, +43 % on Kimi's first three steps alone), and it is still
# at one microbatch here.


def deepseek_v4_flash_8k_sac(seq_len: int | None = 8192) -> Trainer.Config:
    """F6. SelectiveAC at 8k -- the middle rung between FullAC and none.

    SAC recomputes less than FullAC while storing far less than no AC, so it
    should land between 95 GiB and the ~265 GiB no-AC wanted. Prior results are
    genuinely mixed and both are explicable: -10.2 % on Kimi (it cost +28.75
    GiB there and bought nothing), and -85 % on `pro` -- but every `pro` SAC run
    sat at or past the ~90 % memory cliff, so that number measured the cliff,
    not SAC. Here there is room for it to be measured honestly for once.
    """
    config = _flash_8k(seq_len)
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def deepseek_v4_flash_8k_bs2(seq_len: int | None = 8192) -> Trainer.Config:
    """F7. 2x microbatch (16384 tokens/rank), projected ~134 GiB.

    The 8k reference uses 95.14 GiB for 8192 tokens/rank; the token-dependent
    part is ~39 GiB (it was 56.14 GiB at 4096), so doubling should land near
    134 GiB and stay far clear of the retry threshold.
    """
    config = _flash_8k(seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def deepseek_v4_flash_8k_bs3(seq_len: int | None = 8192) -> Trainer.Config:
    """F8. 3x microbatch (24576 tokens/rank), projected ~173 GiB.

    Judge memory at step 6 or later -- peak drifts well past step 2 on this
    cluster -- and check the log for `CUDA memory allocation retries`: any
    nonzero count means the run is in the collapse regime whatever the reported
    peak says.
    """
    config = _flash_8k(seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 24576
    return config


# ===========================================================================
# Round 2: lower parallelism, less AC, bigger batch -- all composed on EP=16.
#
# EP 64 -> 16 measured **21.37 TFLOP/s against the 8k reference's 18.19
# (+17.5 %)** and *freed* 29.4 GiB (95.14 -> 65.70 GiB). Both directions matter:
#
#   * Throughput: EP=16 is now the optimum on a THIRD model on this cluster --
#     `pro` +13.7 % (from 64), Kimi K2.7 +8.8 % (from 8), flash +17.5 % (from
#     64). 4 GPUs/node, no NVSwitch, 4 IB HCAs.
#   * Memory: the AllToAll dispatch workspace scales with `ep_size`, so EP=64's
#     buffers are ~4x EP=16's, and that term dominates the extra expert weight
#     each rank holds (16 experts at EP=16 against 4 at EP=64). Note this is the
#     OPPOSITE direction to Kimi, where higher EP meant lower memory -- same
#     lever, different binding constraint.
#
# Since memory falls as EP falls, the sweep is worth continuing down. The
# counter-pressure is FSDP: at EP=1 all 256 routed experts of a layer are left
# to FSDP, and one all-gather materialises 256 x 3 x 4096 x 2048 x 2 B =
# 12.9 GB for a single layer, more with prefetch. Somewhere between 16 and 1
# that overtakes the dispatch saving, and only a measurement says where.
# ===========================================================================


def _flash_8k_ep(ep: int, seq_len: int | None = 8192) -> Trainer.Config:
    config = _flash_8k(seq_len)
    config.parallelism = ParallelismConfig(expert_parallel_degree=ep)
    return config


def deepseek_v4_flash_8k_ep8(seq_len: int | None = 8192) -> Trainer.Config:
    """F9. EP=8 -- 32 experts/rank, dispatch confined to 2 nodes."""
    return _flash_8k_ep(8, seq_len)


def deepseek_v4_flash_8k_ep4(seq_len: int | None = 8192) -> Trainer.Config:
    """F10. EP=4 -- 64 experts/rank, dispatch stays inside one node (no IB hop).

    The first point where MoE dispatch never leaves the node. On Kimi EP=4 was
    the worst of the sweep (150.95, and 87 % memory), but Kimi's expert weights
    are far larger; flash's 12.9 GB/layer may make this affordable.
    """
    return _flash_8k_ep(4, seq_len)


def deepseek_v4_flash_8k_ep2(seq_len: int | None = 8192) -> Trainer.Config:
    """F11. EP=2 -- 128 experts/rank."""
    return _flash_8k_ep(2, seq_len)


def deepseek_v4_flash_8k_ep1(seq_len: int | None = 8192) -> Trainer.Config:
    """F12. EP=1 -- no expert parallelism at all, the STOCK setting.

    All 256 experts per layer go to FSDP. The flash baseline docstring flags
    the cost: one all-gather materialises 12.9 GB for a single layer, more with
    prefetch. Included because "much lower parallelism" deserves its endpoint
    measured rather than assumed -- and because the config-validation path that
    rejects CUDA graphs returns early at EP=1, so this is also the only EP where
    CUDA graphs would be legal.
    """
    return _flash_8k_ep(1, seq_len)


# --- less AC and more batch, on the EP=16 base -----------------------------


def deepseek_v4_flash_8k_ep16_sac(seq_len: int | None = 8192) -> Trainer.Config:
    """F13. EP=16 + SelectiveAC -- "less AC", made affordable.

    No AC needs ~265 GiB of activations at 8k (F1 OOMed with the GPU at 275.26
    of 276.50), so removing checkpointing entirely is out of reach at this
    context length whatever EP does. SAC is the reachable middle: it recomputes
    less than FullAC while storing far less than nothing, and EP=16's 65.70 GiB
    base leaves ~210 GiB for it to grow into.
    """
    config = _flash_8k_ep(16, seq_len)
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def deepseek_v4_flash_8k_ep16_bs2(seq_len: int | None = 8192) -> Trainer.Config:
    """F14. EP=16 + 2x microbatch (16384 tokens/rank), projected ~105 GiB."""
    config = _flash_8k_ep(16, seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def deepseek_v4_flash_8k_ep16_bs4(seq_len: int | None = 8192) -> Trainer.Config:
    """F15. EP=16 + 4x microbatch (32768 tokens/rank), projected ~183 GiB.

    Batch is the most reliable lever found across all three models here. On
    Kimi it went 195 -> 305 TFLOP/s and did not flatten until bs=8, so jumping
    straight to 4x rather than walking 2x/3x is the better use of a slot -- the
    2x run brackets it from below.
    """
    config = _flash_8k_ep(16, seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 32768
    return config


def deepseek_v4_flash_8k_ep16_sac_bs2(seq_len: int | None = 8192) -> Trainer.Config:
    """F16. All three at once: EP=16 + SelectiveAC + 2x microbatch.

    The combination asked for -- lower parallelism, less AC, bigger batch. Run
    alongside the single-variable configs: if it beats them the changes compose,
    and if not those runs say which is fighting which.
    """
    config = _flash_8k_ep(16, seq_len)
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


# --- the two proven winners, composed --------------------------------------
#
#   8k reference          18.19 TFLOP/s   95.14 GiB
#   EP=16                 21.37 (+17.5 %) 65.70 GiB
#   block_size 32         21.17 (+16.4 %) 75.89 GiB
#
# They attack different costs -- EP is MoE dispatch scope, block_size is
# attention score area -- so they should compose. Both also *free* memory
# (-29.4 and -19.3 GiB), which is what makes the batch variant reachable.


def deepseek_v4_flash_8k_ep16_blk32(seq_len: int | None = 8192) -> Trainer.Config:
    """F17. EP=16 + block_size 32 -- the two measured winners together."""
    config = _flash_8k_ep(16, seq_len)
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.block_size = 32
    return config


def deepseek_v4_flash_8k_ep16_blk32_batch4(
    seq_len: int | None = 8192,
) -> Trainer.Config:
    """F18. Both winners plus 4x microbatch (32768 tokens/rank).

    If the two savings are additive the base lands near 50 GiB, leaving well
    over 200 GiB for the batch to grow into. Batch has been the most reliable
    lever on every model measured on this cluster, and flash is still running
    one microbatch.
    """
    config = deepseek_v4_flash_8k_ep16_blk32(seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 32768
    return config


def deepseek_v4_flash_8k_ep16_blk32_sac(seq_len: int | None = 8192) -> Trainer.Config:
    """F19. Composed winners with SelectiveAC instead of FullAC.

    Less recompute, more activation memory. The composed base should sit near
    50 GiB of the ~276 GiB card, so the trade is affordable here even though
    it was not at the EP=64 / block_size 128 base.
    """
    config = deepseek_v4_flash_8k_ep16_blk32(seq_len)
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def deepseek_v4_flash_8k_ep16_blk32_no_ac(
    seq_len: int | None = 8192,
) -> Trainer.Config:
    """F20. Composed winners with no AC at all.

    no_ac OOMed on the plain 8k base wanting ~265 GiB. The composed base frees
    roughly 49 GiB, which should bring that under the card -- but only if the
    savings are in the activation term rather than beside it, which is exactly
    what this run measures.
    """
    config = deepseek_v4_flash_8k_ep16_blk32(seq_len)
    config.activation_checkpoint = None
    return config


def deepseek_v4_flash_8k_ep16_blk16(seq_len: int | None = 8192) -> Trainer.Config:
    """F21. Composed base with block_size 16 instead of 32.

    128 -> 32 was +15.8 % on its own and composed cleanly with EP=16. This asks
    whether the DSA mask granularity has further to give or has bottomed out.
    Finer blocks cut wasted score area but raise mask overhead and shrink the
    GEMM tiles, so a reversal here is the expected way for the lever to end.
    """
    config = deepseek_v4_flash_8k_ep16_blk32(seq_len)
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.block_size = 16
    return config


def deepseek_v4_flash_8k_ep4_blk32(seq_len: int | None = 8192) -> Trainer.Config:
    """F22. block_size 32 with EP=4 rather than EP=16.

    On the block_size 128 base the EP sweep was nearly flat between 4 and 16
    (EP=16 21.39, EP=8 21.84, EP=4 22.04, EP=1 20.80), so EP=4 leads by ~3 %
    -- barely over the noise floor, but in the same direction twice. Worth one
    run composed with block_size 32 to see if the ordering holds at the top.
    """
    config = _flash_8k_ep(4, seq_len)
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.block_size = 32
    return config


def deepseek_v4_flash_8k_ep16_blk32_batch2(
    seq_len: int | None = 8192,
) -> Trainer.Config:
    """F23. Composed base with 2x microbatch.

    The 4x attempt (job 330) burned the full 3 h wall clock without reaching
    step 1 -- not an OOM, a compile/autotune blowup at 32768 tokens. 2x halves
    the new shape pressure while still testing whether batch pays here at all.
    """
    config = deepseek_v4_flash_8k_ep16_blk32(seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config


def _flash_8k_ep_blk(
    ep: int, block_size: int = 32, seq_len: int | None = 8192
) -> Trainer.Config:
    """EP degree x block_size, the two levers that compose multiplicatively."""
    config = _flash_8k_ep(ep, seq_len)
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.block_size = block_size
    return config


def deepseek_v4_flash_8k_ep8_blk32(seq_len: int | None = 8192) -> Trainer.Config:
    """F24. EP=8 + block_size 32 -- fills in the composed EP curve.

    Composed: EP=16 24.74, EP=4 25.45. On the block_size 128 base EP=8 sat
    between them (21.84 vs 21.39 and 22.04), so this checks the curve is
    monotone in the 4..16 range rather than bumpy.
    """
    return _flash_8k_ep_blk(8, 32, seq_len)


def deepseek_v4_flash_8k_ep2_blk32(seq_len: int | None = 8192) -> Trainer.Config:
    """F25. EP=2 + block_size 32 -- is EP=4 the floor or just the lowest tested?

    EP=1 was clearly bad (20.80, and 128.98 GiB because every rank then holds
    all 256 routed experts for FSDP to all-gather). EP=2 is the untested rung
    between that collapse and the EP=4 lead, so it decides whether 4 is a real
    optimum or simply the edge of the sweep.
    """
    return _flash_8k_ep_blk(2, 32, seq_len)


def deepseek_v4_flash_8k_ep4_blk32_profile(
    seq_len: int | None = 8192,
) -> Trainer.Config:
    """The 8k round's WINNER, profiled. Companion to the Pro profile pair.

    Identical to `deepseek_v4_flash_8k_ep4_blk32` -- EP=4, FlexAttention
    `block_size` 32, bf16, FullAC, `disable_cuda_graphs` -- with the torch
    profiler turned on for two steps. Measured 25.45 TFLOP/s per GPU without
    profiling (30 steps, mean over steps 15+, peak 76.58 GiB), which is
    +39.1 % over the 8k reference and the best of the round.

    Why this config and not the 4096 baseline: `pro` already has a
    before/after profile pair, and the open question for `flash` is not "what
    did the optimization round change" but "what is the remaining 25.45
    TFLOP/s made of". At 27.7 % of HBM and with attention still on 32x32
    Triton tiles, the kernel breakdown is the only thing that says whether the
    next lever is attention, MoE dispatch, or something not yet suspected.

    `profiler_warmup=3, profiler_active=2` matches
    `deepseek_v4_pro_64xgb300_baseline_profile` so the two traces are directly
    comparable: past compile and autotune, small enough to open. Profiling
    distorts step time, so numbers from inside a profiled run are not
    throughput datapoints -- the kernel breakdown is the product.
    """
    config = deepseek_v4_flash_8k_ep4_blk32(seq_len=seq_len)
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=10,
        profiler_warmup=3,
        profiler_active=2,
    )
    return config


def deepseek_v4_flash_8k_ep4_blk32_batch2(
    seq_len: int | None = 8192,
) -> Trainer.Config:
    """F26. The best config (EP=4 + block_size 32) at 2x microbatch.

    Batch has never actually been measured on flash: the two earlier attempts
    were built on the EP=16 base before EP=4 won, and neither reached step 1.
    4x (job 330) hit the 3 h wall clock and 2x (job 341) was cancelled after
    30 min of silence, so both are non-results rather than verdicts.

    Memory is not the constraint. The best config peaks at 76.58 GiB of the
    ~276.5 GiB card, leaving ~200 GiB, and under FullAC the *stored* activation
    term is small by construction -- doubling the microbatch should cost tens of
    GiB, not hundreds.

    The real cost is startup: a new microbatch shape re-benchmarks every flex
    kernel, ~18 s x 64 blocks = ~19 min before step 1, on top of normal init.
    Run this with a wall clock well past 3 h so that cost cannot be mistaken
    for a hang again.
    """
    config = _flash_8k_ep_blk(4, 32, seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    return config
