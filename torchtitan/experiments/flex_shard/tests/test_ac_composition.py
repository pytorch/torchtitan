#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test composition of activation checkpointing + FlexShard reshard checkpoint.

Verifies that when both AC and FlexShard are applied:
1. Each layer has exactly one CheckpointWrapper (not nested)
2. FlexShard collective ops are marked MUST_RECOMPUTE (reshard semantics)
3. Forward/backward produces correct numerics vs. unsharded reference

Usage:
    torchrun --nproc_per_node=2 \
      torchtitan/experiments/flex_shard/tests/test_ac_composition.py
"""

import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    per_param_placements,
)
from torchtitan.experiments.flex_shard.sharding_metadata import (
    _EAGER_COMM_CONTEXTS_ATTR,
)


class SimpleMLP(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for _ in range(3)]
        )
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)


def apply_selective_ac(model):
    """Apply selective AC to each layer (simulates torchtitan's apply_ac)."""
    save_ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
    }

    def _policy(ctx, func, *args, **kwargs):
        from torch.utils.checkpoint import CheckpointPolicy

        if func in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    for i in range(len(model.layers)):
        model.layers[i] = checkpoint_wrapper(
            model.layers[i],
            context_fn=lambda: create_selective_checkpoint_contexts(_policy),
        )


def apply_full_ac(model):
    """Apply vanilla/full AC to each layer."""
    for i in range(len(model.layers)):
        model.layers[i] = checkpoint_wrapper(model.layers[i])


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def _init_flex_mesh(world_size):
    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))


def _flex_shard(model, mesh, **kwargs):
    kwargs.setdefault("buckets", _default_bucket_specs(model))
    return flex_shard(model, mesh, DataParallelMeshDims(shard="fsdp"), **kwargs)


def _default_bucket_specs(model):
    bucket_specs = []
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList):
            bucket_specs.extend(
                BucketSpec([f"{name}.{idx}.*"]) for idx in range(len(child))
            )
        else:
            bucket_specs.append(BucketSpec([f"{name}.*"]))
    return bucket_specs


def _transformer_bucket_specs(num_layers):
    return (
        [
            BucketSpec(["tok_embeddings.*"], reshard_after_forward=True),
            BucketSpec(["pos_embeddings.*"], reshard_after_forward=True),
        ]
        + [
            BucketSpec([f"layers.{idx}.*"], reshard_after_forward=True)
            for idx in range(num_layers)
        ]
        + [
            BucketSpec(["norm.*"], reshard_after_forward=True),
            BucketSpec(["output.*"], reshard_after_forward=True),
        ]
    )


def _unwrap_checkpoint(module):
    while hasattr(module, "_checkpoint_wrapped_module"):
        module = module._checkpoint_wrapped_module
    return module


def _clean_checkpoint_fqn(fqn):
    return fqn.replace("._checkpoint_wrapped_module", "")


def _expected_chunk(tensor, chunks, dim, rank):
    result = list(torch.chunk(tensor, chunks, dim=dim))
    empty_shape = list(tensor.shape)
    empty_shape[dim] = 0
    while len(result) < chunks:
        result.append(tensor.new_empty(empty_shape))
    return result[rank].contiguous()


def test_no_nested_wrappers():
    """AC + FlexShard produces a single CheckpointWrapper per layer, not nested."""
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)

    torch.manual_seed(42)
    model = SimpleMLP().cuda()
    dist.broadcast(model.layers[0][0].weight.data, src=0)

    # Apply AC first (as torchtitan does)
    apply_selective_ac(model)

    # Verify AC wrapped
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not AC-wrapped"

    # Apply FlexShard with reshard_after_forward
    _flex_shard(model, mesh, shard_placement_fn=per_param_placements)

    # Verify: single wrapper, not nested
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not wrapped"
        inner = layer._checkpoint_wrapped_module
        assert not isinstance(inner, CheckpointWrapper), (
            f"layers.{i} has nested CheckpointWrapper"
        )

    print_rank0("PASSED: no_nested_wrappers")


def test_numerics_ac_plus_flexshard():
    """AC + FlexShard produces correct loss matching DDP reference."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)
    device = torch.device("cuda", torch.cuda.current_device())

    # FlexShard + AC model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = SimpleMLP().to(device)
    apply_selective_ac(model)
    _flex_shard(model, mesh, shard_placement_fn=per_param_placements)

    # DDP reference
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = DDP(SimpleMLP().to(device), device_ids=[rank])

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for step in range(3):
        torch.manual_seed(42 + rank + step)
        inp = torch.randn(4, 16, device=device)

        opt.zero_grad()
        loss = model(inp).sum()
        loss.backward()
        opt.step()

        ref_opt.zero_grad()
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        ref_opt.step()

        torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-4)
        if rank == 0:
            print(
                f"  step {step}: flex+ac={loss.item():.6f}  ref={ref_loss.item():.6f}"
            )

    print_rank0("PASSED: numerics_ac_plus_flexshard")


def test_flexshard_only():
    """FlexShard without AC still works (no AC wrapper to detect)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)
    device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = SimpleMLP().to(device)
    _flex_shard(model, mesh, shard_placement_fn=per_param_placements)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = DDP(SimpleMLP().to(device), device_ids=[rank])

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    torch.manual_seed(42 + rank)
    inp = torch.randn(4, 16, device=device)

    opt.zero_grad()
    loss = model(inp).sum()
    loss.backward()
    opt.step()

    ref_opt.zero_grad()
    ref_loss = ref_model(inp).sum()
    ref_loss.backward()
    ref_opt.step()

    torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-4)
    print_rank0("PASSED: flexshard_only")


def _run_transformer_checkpoint_raf_recompute_prefetch(apply_ac_fn, label):
    """Transformer AC + RAF uses phase-correct pending AG during recompute."""
    import importlib

    reshard_mod = importlib.import_module(
        "torchtitan.experiments.flex_shard.reshard_after_forward"
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)
    device = torch.device("cuda", torch.cuda.current_device())
    model_args = ModelArgs(
        n_layers=2,
        vocab_size=32,
        max_seq_len=8,
        dim=16,
        n_heads=4,
        dropout_p=0.0,
        use_attn_mask=True,
        weight_tying=False,
        checkpoint_activations=False,
    )

    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)
    model = Transformer(model_args).to(device)
    apply_ac_fn(model)
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not AC-wrapped"
    bucket_specs = _transformer_bucket_specs(model_args.n_layers)
    _flex_shard(
        model,
        mesh,
        shard_placement_fn=per_param_placements,
        buckets=bucket_specs,
    )

    context = next(iter(getattr(model, _EAGER_COMM_CONTEXTS_ATTR).values()))
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not wrapped"
        assert not isinstance(
            layer._checkpoint_wrapped_module,
            CheckpointWrapper,
        ), f"layers.{i} has nested CheckpointWrapper"

    bucket_names = [bucket.debug_fqn for bucket in context.buckets]
    expected_clean_order = [
        "tok_embeddings",
        "pos_embeddings",
        "layers.0",
        "layers.1",
        "norm",
        "output",
    ]
    assert [_clean_checkpoint_fqn(name) for name in bucket_names] == (
        expected_clean_order
    ), bucket_names

    records = []

    def _make_probe(name):
        def _probe(_mod, _args):
            pending = context.pending
            records.append(
                {
                    "phase": (
                        "recompute"
                        if reshard_mod._reshard_after_forward_recompute.get()
                        else "forward"
                    ),
                    "bucket": name,
                    "pending": (
                        pending.bucket.debug_fqn if pending is not None else None
                    ),
                    "pending_recompute": (
                        pending.recompute if pending is not None else None
                    ),
                }
            )

        return _probe

    probe_targets_by_clean_name = {
        "tok_embeddings": _unwrap_checkpoint(model.tok_embeddings),
        "pos_embeddings": _unwrap_checkpoint(model.pos_embeddings),
        "layers.0": _unwrap_checkpoint(model.layers[0]),
        "layers.1": _unwrap_checkpoint(model.layers[1]),
        "norm": _unwrap_checkpoint(model.norm),
        "output": _unwrap_checkpoint(model.output),
    }
    for name in bucket_names:
        clean_name = _clean_checkpoint_fqn(name)
        probe_targets_by_clean_name[clean_name].register_forward_pre_hook(
            _make_probe(name)
        )

    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)
    ref_model = Transformer(model_args).to(device)

    torch.manual_seed(42)
    inp = torch.randint(
        0,
        model_args.vocab_size,
        (2, model_args.max_seq_len),
        device=device,
    )
    dist.broadcast(inp, src=0)

    loss = model(inp).square().mean()
    ref_loss = ref_model(inp).square().mean()
    torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-4)

    loss.backward()
    ref_loss.backward()

    forward_records = [record for record in records if record["phase"] == "forward"]
    recompute_records = [record for record in records if record["phase"] == "recompute"]
    assert [record["bucket"] for record in forward_records] == bucket_names, records
    assert [record["bucket"] for record in recompute_records] == list(
        reversed(bucket_names)
    ), records

    expected_forward_pending = dict(
        zip(bucket_names, bucket_names[1:] + [None], strict=True)
    )
    expected_recompute_order = list(reversed(bucket_names))
    expected_recompute_pending = dict(
        zip(
            expected_recompute_order,
            expected_recompute_order[1:] + [None],
            strict=True,
        )
    )
    for record in forward_records:
        assert record["pending"] == expected_forward_pending[record["bucket"]]
        if record["pending"] is not None:
            assert record["pending_recompute"] is False
    for record in recompute_records:
        assert record["pending"] == expected_recompute_pending[record["bucket"]]
        if record["pending"] is not None:
            assert record["pending_recompute"] is True
    assert context.pending is None

    ref_grads = {
        name: param.grad.detach()
        for name, param in ref_model.named_parameters()
        if param.grad is not None
    }
    for bucket in context.buckets:
        for leaf, name, _param_p, info in bucket.entries:
            param = leaf._parameters[name]
            ref_fqn = _clean_checkpoint_fqn(info.fqn)
            expected_grad = _expected_chunk(
                ref_grads[ref_fqn],
                world_size,
                0,
                rank,
            )
            torch.testing.assert_close(param.grad, expected_grad, atol=1e-5, rtol=1e-4)

    print_rank0(f"PASSED: transformer_checkpoint_raf_recompute_prefetch_{label}")


def test_transformer_selective_ac_checkpoint_raf_recompute_prefetch():
    _run_transformer_checkpoint_raf_recompute_prefetch(
        apply_selective_ac,
        "selective_ac",
    )


def test_transformer_full_ac_checkpoint_raf_recompute_prefetch():
    _run_transformer_checkpoint_raf_recompute_prefetch(
        apply_full_ac,
        "full_ac",
    )


def main():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    tests = [
        test_no_nested_wrappers,
        test_numerics_ac_plus_flexshard,
        test_flexshard_only,
        test_transformer_selective_ac_checkpoint_raf_recompute_prefetch,
        test_transformer_full_ac_checkpoint_raf_recompute_prefetch,
    ]

    success = True
    for test in tests:
        try:
            test()
        except Exception as e:
            print_rank0(f"FAILED: {test.__name__}: {e}")
            traceback.print_exc()
            success = False

    dist.barrier()
    rank = dist.get_rank()
    dist.destroy_process_group()

    if not success:
        raise SystemExit(1)
    if rank == 0:
        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
