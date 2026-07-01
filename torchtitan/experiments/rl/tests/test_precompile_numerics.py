# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-GPU bitwise numerics check for the RL aot_fx_trace path.

The full RL loop needs Monarch + TorchStore + vLLM, which are not required to
validate the piece this PR adds: the traced forward+loss+backward step. This
script builds the RL qwen3 model under SimpleFSDP (dp_shard=1) and asserts that
``RLTracedStep`` (trace + regional-Inductor + run) reproduces the eager
ChunkedLoss + GRPO forward/backward *bitwise* (loss, grads, metrics), which is
the correctness gate for the change.

It is written as a standalone script (world_size=1) rather than a pytest case
because it initializes a real process group. Run directly:

    python -m torchtitan.experiments.rl.tests.test_precompile_numerics
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.rl.actors.traced_step import RLTracedStep
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.simple_fsdp_parallelize import to_simple_fsdp_spec
from torchtitan.experiments.rl.precompile import save_rl_precompiled
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.tools import utils

FLAVOR = "debugmodel"
SEQ_LEN = 256
BATCH = 2
NUM_CHUNKS = 8
NUM_DOCS_PER_ROW = 4  # emulate packed multi-document rows


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


def _make_config(seq_len: int) -> SimpleNamespace:
    """A config-like object for update_from_config / parallelize / SimpleFSDP."""
    return SimpleNamespace(
        parallelism=ParallelismConfig(),
        training=TrainingConfig(
            seq_len=seq_len,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        override=SimpleNamespace(imports=[]),
    )


def build_model(device: torch.device, parallel_dims: ParallelDims, config):
    """Mirror PolicyTrainer._build_model for the SimpleFSDP aot_fx_trace path."""
    import dataclasses

    spec = qwen3_model_registry(
        FLAVOR, attn_backend="flex", converters=[LMHeadCastConverter.Config()]
    )
    spec = to_simple_fsdp_spec(spec)

    spec.model.update_from_config(config=config)
    # RoPE cache is sized to training.seq_len (the non-trainer update_from_config
    # path skips this, so do it explicitly as _build_model does).
    for layer_cfg in spec.model.layers:
        attn = getattr(layer_cfg, "attention", None)
        if attn is not None:
            attn.rope = dataclasses.replace(
                attn.rope, max_seq_len=config.training.seq_len
            )

    with torch.device("meta"):
        with utils.set_default_dtype(torch.float32):
            model = spec.model.build()

    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=GraphTrainerCompileConfig(enable=True, mode="aot_fx_trace"),
        ac_config=None,
        dump_folder="",
    )
    model.to_empty(device=device.type)
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    model.train()
    return model, spec


def make_inputs(device: torch.device, vocab_size: int):
    torch.manual_seed(1234)
    token_ids = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=device)
    labels = torch.randint(0, vocab_size, (BATCH, SEQ_LEN), device=device)
    # Emulate packed multi-document rows: positions reset to 0 at each document
    # boundary (this is what makes the flex BlockMask content batch-dependent).
    doc_len = SEQ_LEN // NUM_DOCS_PER_ROW
    positions = (
        torch.arange(doc_len, dtype=torch.int32, device=device)
        .repeat(NUM_DOCS_PER_ROW)
        .expand(BATCH, SEQ_LEN)
        .contiguous()
    )
    generator_logprobs = torch.randn(BATCH, SEQ_LEN, device=device) * 0.1 - 2.0
    advantages = torch.randn(BATCH, SEQ_LEN, device=device)
    loss_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool, device=device)
    loss_mask[:, doc_len // 2 :] = True  # some response tokens
    advantages = advantages * loss_mask
    return token_ids, labels, positions, generator_logprobs, advantages, loss_mask


def snapshot_grads(model) -> dict[str, torch.Tensor]:
    return {
        n: _full(p.grad).detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def zero_grads(model) -> None:
    for p in model.parameters():
        p.grad = None


def eager_forward_backward(model, loss_fn, inputs, gvt):
    token_ids, labels, positions, gen_lp, adv, loss_mask = inputs
    attention_masks = model.get_attention_masks(positions)
    pred = model(token_ids, attention_masks=attention_masks, positions=positions)
    loss, metrics = loss_fn(
        pred,
        labels,
        gvt,
        generator_logprobs=gen_lp,
        advantages=adv,
        loss_mask=loss_mask,
    )
    loss.backward()
    return loss.detach().clone(), {k: v.detach().clone() for k, v in metrics.items()}


def build_loss(loss_cls, model):
    loss_fn = loss_cls.Config(num_chunks=NUM_CHUNKS, loss_fn=GRPOLoss.Config()).build()
    loss_fn.set_lm_head(model.lm_head)
    model._skip_lm_head = True
    return loss_fn


def assert_bitwise(name, a, b):
    a, b = _full(a), _full(b)
    if not torch.equal(a, b):
        max_abs = (a.float() - b.float()).abs().max().item()
        raise AssertionError(f"{name}: NOT bitwise equal, max|diff|={max_abs:.3e}")
    print(f"  [OK] {name}: bitwise equal")


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=0, world_size=1)

    try:
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
        )
        parallel_dims.build_mesh()

        config = _make_config(SEQ_LEN)
        model, spec = build_model(device, parallel_dims, config)
        vocab_size = spec.model.vocab_size
        inputs = make_inputs(device, vocab_size)
        int_gvt = int(inputs[5].sum().item())
        gvt_tensor = torch.tensor(float(int_gvt), dtype=torch.float32, device=device)
        print(
            f"Model={FLAVOR} seq_len={SEQ_LEN} batch={BATCH} "
            f"num_chunks={NUM_CHUNKS} gvt={int_gvt}"
        )

        # 1) Eager reference: ChunkedLossWrapperWithParamGrads + loss.backward.
        zero_grads(model)
        loss_ref_fn = build_loss(ChunkedLossWrapperWithParamGrads, model)
        loss_ref, metrics_ref = eager_forward_backward(
            model, loss_ref_fn, inputs, gvt_tensor
        )
        grads_ref = snapshot_grads(model)
        print(f"eager(ParamGrads) loss={loss_ref.item():.6f} #grads={len(grads_ref)}")

        # 2) Traced path (RLTracedStep) on the same model + weights.
        zero_grads(model)
        traced_fn = build_loss(ChunkedLossWrapperWithParamGrads, model)
        traced_step = RLTracedStep(
            compile_config=GraphTrainerCompileConfig(enable=True, mode="aot_fx_trace"),
            parallel_dims=parallel_dims,
            parallelism=config.parallelism,
            model_config=spec.model,
            loss_config=ChunkedLossWrapper.Config(
                num_chunks=NUM_CHUNKS, loss_fn=GRPOLoss.Config()
            ),
            train_context=dist_utils.get_spmd_context(
                parallel_dims=parallel_dims, spmd_typechecking=False
            ),
            dump_folder="",
        )
        tk, lb, pos, gen_lp, adv, lm = inputs
        loss_tr, metrics_tr = traced_step.run(
            model,
            traced_fn,
            token_ids=tk,
            labels=lb,
            positions=pos,
            attention_masks=model.get_attention_masks(pos),
            generator_logprobs=gen_lp,
            advantages=adv,
            loss_mask=lm,
            global_valid_tokens=int_gvt,
            device=device,
        )
        grads_tr = snapshot_grads(model)
        # Under cudagraph the returned loss/metrics alias static replay buffers
        # that the next run() overwrites, so copy them out before comparing.
        loss_tr = loss_tr.detach().clone()
        metrics_tr = {k: v.detach().clone() for k, v in metrics_tr.items()}
        print(f"traced          loss={loss_tr.item():.6f} #grads={len(grads_tr)}")

        # ---- Primary gate: traced == eager, bitwise ----
        print("Comparing traced vs eager(ParamGrads):")
        assert_bitwise("loss", loss_ref, loss_tr)
        assert set(grads_ref) == set(grads_tr), "grad key mismatch"
        for name in sorted(grads_ref):
            assert_bitwise(f"grad[{name}]", grads_ref[name], grads_tr[name])
        for k in sorted(metrics_ref):
            assert_bitwise(f"metric[{k}]", metrics_ref[k], metrics_tr[k])

        # 3) The base ChunkedLossWrapper (production eager) must give the same
        #    grads as the ParamGrads variant used by the traced path.
        print("Comparing base ChunkedLossWrapper vs ParamGrads (eager):")
        zero_grads(model)
        base_fn = build_loss(ChunkedLossWrapper, model)
        loss_base, _ = eager_forward_backward(model, base_fn, inputs, gvt_tensor)
        grads_base = snapshot_grads(model)
        assert_bitwise("loss(base vs paramgrads)", loss_ref, loss_base)
        for name in sorted(grads_ref):
            assert_bitwise(f"grad(base)[{name}]", grads_ref[name], grads_base[name])

        # 4) global_valid_tokens must be a live graph input, not a baked
        #    constant: same inputs, different gvt -> loss scales by the ratio.
        #    Copy each scalar out immediately (cudagraph reuses the output buffer
        #    across replays).
        print("Checking gvt is a live input (not baked):")
        zero_grads(model)
        loss_a = float(
            traced_step.run(
                model,
                traced_fn,
                token_ids=tk,
                labels=lb,
                positions=pos,
                attention_masks=model.get_attention_masks(pos),
                generator_logprobs=gen_lp,
                advantages=adv,
                loss_mask=lm,
                global_valid_tokens=int_gvt,
                device=device,
            )[0]
        )
        zero_grads(model)
        loss_b = float(
            traced_step.run(
                model,
                traced_fn,
                token_ids=tk,
                labels=lb,
                positions=pos,
                attention_masks=model.get_attention_masks(pos),
                generator_logprobs=gen_lp,
                advantages=adv,
                loss_mask=lm,
                global_valid_tokens=2 * int_gvt,
                device=device,
            )[0]
        )
        ratio = loss_a / loss_b
        assert abs(ratio - 2.0) < 1e-3, f"gvt not live: loss ratio={ratio} (want ~2.0)"
        print(f"  [OK] doubling gvt halved the loss (ratio={ratio:.5f})")

        # 5) CooR precompile roundtrip: save an artifact, then load it in a
        #    fresh RLTracedStep and confirm the loaded graph reproduces the
        #    eager grads bitwise (validates serialization + load + cudagraph
        #    re-capture at load time).
        print("CooR precompile save/load roundtrip:")
        with tempfile.TemporaryDirectory() as artifact_dir:
            zero_grads(model)
            save_cfg = GraphTrainerCompileConfig(
                enable=True, mode="aot_fx_trace", precompile_artifact_dir=artifact_dir
            )
            save_loss = build_loss(ChunkedLossWrapperWithParamGrads, model)
            save_rl_precompiled(
                model,
                save_loss,
                compile_config=save_cfg,
                parallel_dims=parallel_dims,
                parallelism=config.parallelism,
                model_config=spec.model,
                loss_config=ChunkedLossWrapper.Config(
                    num_chunks=NUM_CHUNKS, loss_fn=GRPOLoss.Config()
                ),
                train_context=dist_utils.get_spmd_context(
                    parallel_dims=parallel_dims, spmd_typechecking=False
                ),
                token_ids=tk,
                labels=lb,
                positions=pos,
                attention_masks=model.get_attention_masks(pos),
                generator_logprobs=gen_lp,
                advantages=adv,
                loss_mask=lm,
                global_valid_tokens=int_gvt,
                device=device,
            )
            zero_grads(model)
            load_step = RLTracedStep(
                compile_config=GraphTrainerCompileConfig(
                    enable=True,
                    mode="aot_fx_trace",
                    precompile_artifact_dir=artifact_dir,
                ),
                parallel_dims=parallel_dims,
                parallelism=config.parallelism,
                model_config=spec.model,
                loss_config=ChunkedLossWrapper.Config(
                    num_chunks=NUM_CHUNKS, loss_fn=GRPOLoss.Config()
                ),
                train_context=dist_utils.get_spmd_context(
                    parallel_dims=parallel_dims, spmd_typechecking=False
                ),
                dump_folder="",
            )
            load_loss = build_loss(ChunkedLossWrapperWithParamGrads, model)
            loss_pc, _ = load_step.run(
                model,
                load_loss,
                token_ids=tk,
                labels=lb,
                positions=pos,
                attention_masks=model.get_attention_masks(pos),
                generator_logprobs=gen_lp,
                advantages=adv,
                loss_mask=lm,
                global_valid_tokens=int_gvt,
                device=device,
            )
            loss_pc = loss_pc.detach().clone()
            grads_pc = snapshot_grads(model)
            assert_bitwise("loss(precompiled vs eager)", loss_ref, loss_pc)
            for name in sorted(grads_ref):
                assert_bitwise(
                    f"grad(precompiled)[{name}]", grads_ref[name], grads_pc[name]
                )

        print("\nALL CHECKS PASSED")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
