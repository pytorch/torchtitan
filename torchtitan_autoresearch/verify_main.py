"""Tier-1 single-batch gradient probe (the strong, dataset-robust check).

Launched with the *same* torchrun line as a candidate. It builds the candidate
model through TorchTitan's own ``config.build()`` (so the agent's edited
``parallelize_qwen3`` runs exactly), executes one deterministic forward/backward
on a fixed batch, and either captures a golden/champion snapshot or compares the
candidate against one. The optimizer is never stepped and RNG is left untouched,
so this never mutates training state.

Usage (via run_verify.sh which mirrors run_train.sh):

    capture:  --verify.mode=capture --verify.snapshot=goldens/seq128_bf16.pt
    compare:  --verify.mode=compare --verify.snapshot=goldens/seq128_bf16.pt \
              --verify.champion=goldens/champion.pt --verify.cls=precision
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import ConfigManager
from torchtitan.distributed import utils as dist_utils

from torchtitan_autoresearch import compare as cmp
from torchtitan_autoresearch.verify_config import split_verify_args


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _fingerprint(config) -> dict:
    """Shape/model identity embedded in every snapshot.

    Comparing across a mismatched fingerprint is refused in compare(), which is
    what stops a shape-dependent kernel constraint (e.g. MXFP8's 128-row dim1
    tiling) from silently passing a verify run captured at a different shape.
    """
    return {
        "module": getattr(config, "model_name", "qwen3"),
        "config": getattr(config, "config_name", "qwen3_14b"),
        "seq_len": int(config.training.seq_len),
        "local_batch_size": int(config.training.local_batch_size),
    }


def _fixed_batch(trainer, kind: str, seed: int, device: torch.device):
    """Produce a deterministic (input_dict, labels) on ``device``.

    ``data`` takes c4_test batch 0 under the seeded dataloader; ``stress`` builds
    a synthetic full-vocab batch to probe input regions c4_test never covers.
    Both are reproducible across capture and compare, which is all the
    differential check needs.
    """
    if kind == "data":
        gen = trainer.batch_generator(trainer.dataloader)
        input_dict, labels = next(gen)
        input_dict = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in input_dict.items()
        }
        return input_dict, labels.to(device)

    g = torch.Generator(device="cpu").manual_seed(seed)
    b = int(trainer.config.training.local_batch_size)
    s = int(trainer.config.training.seq_len)
    vocab = int(trainer.tokenizer.get_vocab_size())
    tokens = torch.randint(0, vocab, (b, s), generator=g)
    labels = torch.randint(0, vocab, (b, s), generator=g)
    return {"input": tokens.to(device)}, labels.to(device)


def _global_valid_tokens(trainer, labels: torch.Tensor) -> torch.Tensor:
    pd = trainer.parallel_dims
    local = (labels != IGNORE_INDEX).sum().to(trainer.device)
    if pd.dp_enabled:
        return dist_utils.dist_sum(local, pd.get_mesh("batch"))
    return local.float()


def _to_full(t: torch.Tensor) -> torch.Tensor:
    # full_tensor() is collective: all ranks must call it, every rank then holds
    # the replicated tensor. Plain tensors (FSDP-only baseline) pass through.
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def build_snapshot(trainer, vargs) -> dict:
    """One deterministic fwd/bwd; reduce logits + selected grads to a snapshot."""
    pd = trainer.parallel_dims
    if pd.pp_enabled:
        raise RuntimeError("tier-1 probe does not support pipeline parallel")

    model = trainer.model_parts[0]
    device = trainer.device
    input_dict, labels = _fixed_batch(trainer, vargs.batch, vargs.seed, device)
    gvt = _global_valid_tokens(trainer, labels)

    inputs, labels, extra_inputs, extra_kwargs = trainer.post_dataloading_process(
        input_dict, labels
    )
    # Replicates trainer.forward_backward_step's non-PP branch, but keeps `pred`
    # so we can sample logits (the trainer deletes it before returning).
    with trainer.train_context():
        pred = model(inputs, **extra_inputs, **extra_kwargs)
        full_logits = _to_full(pred).reshape(-1)
        n = full_logits.numel()
        idx = (
            torch.arange(vargs.n_logit_samples, device=device, dtype=torch.int64)
            * cmp._P_BUCKET
            + vargs.seed
        ) % n
        logit_sample = full_logits.index_select(0, idx).detach().to("cpu")
        loss = trainer.loss_fn(pred, labels, gvt)
        del pred, full_logits
        loss.backward()

    # max_norm=inf computes the true distributed grad norm without scaling grads,
    # so the gradients we sketch below are the untouched backward result.
    grad_norm = dist_utils.clip_grad_norm_(
        [p for p in model.parameters()],
        float("inf"),
        foreach=True,
        pp_mesh=None,
        ep_enabled=pd.ep_enabled,
    )

    named = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    selected = cmp.select_param_names(named, vargs.param_patterns)
    grads: dict[str, torch.Tensor] = {}
    for name, p in named:
        if name not in selected:
            continue
        full_grad = _to_full(p.grad)
        grads[name] = cmp.sketch(full_grad, vargs.proj_dim, vargs.seed).to("cpu")

    return {
        "fingerprint": _fingerprint(trainer.config),
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm.item()),
        "logit_sample": logit_sample,
        "grads": grads,
        "selected": selected,
    }


def main() -> int:
    vargs, titan_argv = split_verify_args(sys.argv[1:])
    config = ConfigManager().parse_args(titan_argv)
    # Stash the registry names so the fingerprint is stable across reruns.
    config.model_name = next(
        (a.split("=", 1)[1] for a in titan_argv if a.startswith("--module=")),
        os.environ.get("MODULE", "qwen3"),
    )
    config.config_name = os.environ.get("CONFIG", "qwen3_14b")

    trainer = config.build()  # runs Trainer.__init__ => candidate parallelize_qwen3
    snap = build_snapshot(trainer, vargs)

    code = 0
    if _is_rank0():
        if vargs.mode == "capture":
            torch.save(snap, vargs.snapshot)
            print(f"VERIFY: captured snapshot -> {vargs.snapshot}", flush=True)
        else:
            golden = torch.load(vargs.snapshot, weights_only=False)
            champion = (
                torch.load(vargs.champion, weights_only=False)
                if vargs.champion and os.path.exists(vargs.champion)
                else None
            )
            passed, metrics = cmp.compare(snap, golden, champion, vargs.cls)
            line = " ".join(
                f"{k}={v}"
                for k, v in {
                    "status": "pass" if passed else "FAIL",
                    "cls": metrics.get("cls"),
                    "anchor": metrics.get("anchor"),
                    "logit_sqnr": metrics.get("vs_golden_sqnr_db"),
                    "grad_cosine_min": metrics.get("grad_cosine_min"),
                    "loss_relerr": metrics.get("loss_relerr"),
                }.items()
            )
            print(f"VERIFY: {line}", flush=True)
            if vargs.result_json:
                with open(vargs.result_json, "w") as f:
                    json.dump(metrics, f, indent=2)
            code = 0 if passed else 1

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return code


if __name__ == "__main__":
    sys.exit(main())
