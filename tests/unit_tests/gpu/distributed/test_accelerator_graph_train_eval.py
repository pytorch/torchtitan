# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleSingle,
    ScheduleInterleaved1F1B,
)

from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_manager,
    accelerator_graph_teardown,
    wrap_fwd_bwd_with_accelerator_graph,
    wrap_with_accelerator_graph,
)
from torchtitan.distributed.pipeline_parallel import (
    prepare_pipeline_schedule_for_cuda_graph,
)
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.muse_glimmer import build_model_config


@pytest.fixture(autouse=True)
def reset_cuda_graph_manager():
    accelerator_graph_teardown()
    yield
    accelerator_graph_teardown()


def _validator(*, schedule=None, first=True, last=True):
    validator = object.__new__(Validator)
    validator.config = SimpleNamespace(steps=1)
    validator._cuda_graph_enabled = True
    validator._run_eval = None
    validator._cuda_graph_annotation_config = None
    validator.parallelism = SimpleNamespace(
        num_pp_microbatches=4 if schedule is not None else 1
    )
    validator.parallelism_context = SimpleNamespace(
        pp_enabled=schedule is not None,
        dp_enabled=False,
        dp_cp_enabled=False,
        activate_spmd=lambda: nullcontext(),
    )
    validator.dl_config = SimpleNamespace(max_num_documents=None)
    validator.dp_world_size = 1
    validator.dp_rank = 0
    validator.tokenizer = None
    validator.seq_len = 8
    validator.num_tokens_per_microbatch = 8
    validator.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0, reset=lambda: None
    )
    validator.pp_schedule = schedule
    validator.pp_has_first_stage = first
    validator.pp_has_last_stage = last
    validator.loss_fn = lambda output, target: ((output - target).square().sum(), {})
    return validator


class _EvalLoader(list):
    def close(self):
        pass


def _validate_loss(validator, model_parts, prepared, *, step):
    rows = [
        SimpleNamespace(
            labels=item[1],
            loss_token_counts=torch.ones(1, dtype=torch.int64),
            to_input_dict=lambda device, item=item: {"prepared": item},
        )
        for item in prepared
    ]
    validator.dl_config.build = lambda **kwargs: _EvalLoader(rows)
    losses = []
    validator.metrics_processor.log_validation = lambda *, loss, step: losses.append(
        loss
    )
    with patch.object(
        model_parts[0],
        "preprocess_inputs",
        create=True,
        side_effect=lambda row, **kwargs: (*row["prepared"], {}),
    ):
        validator.validate(model_parts, step)
    return torch.tensor(losses[0] * len(prepared), device="cuda")


@pytest.mark.parametrize("optimizer_graph", [False, True])
def test_train_eval_optimizer_shared_pool(optimizer_graph, monkeypatch):
    torch.manual_seed(42)
    model = torch.nn.Linear(16, 16, device="cuda")
    reference = torch.nn.Linear(16, 16, device="cuda")
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, fused=True, capturable=optimizer_graph
    )
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, fused=True)
    accelerator_graph_manager.capture_allowed = False
    validator = _validator()

    def body(inputs, target):
        loss = (model(inputs) - target).square().sum()
        loss.backward()
        return loss.detach()

    train = wrap_fwd_bwd_with_accelerator_graph(
        body,
        parameters=model.parameters(),
        num_warmup_iterations=2,
        name="train",
    )
    update = (
        wrap_with_accelerator_graph(
            optimizer.step, num_warmup_iterations=2, name="optimizer"
        )
        if optimizer_graph
        else optimizer.step
    )
    addresses = None
    captures = []

    def record_capture(name, boundary):
        if boundary == "after_capture":
            captures.append((step, name))

    monkeypatch.setattr(accelerator_graph_manager, "log_memory", record_capture)
    try:
        for step in range(10):
            inputs = torch.full((4, 16), (step + 1) / 100, device="cuda")
            target = torch.full_like(inputs, 0.05)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected = (reference(inputs) - target).square().sum()
            expected.backward()
            actual = train(inputs, target).clone()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for parameter, expected_parameter in zip(
                model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    parameter.grad, expected_parameter.grad, rtol=0, atol=0
                )
            if step >= 2:
                current = [p.grad.data_ptr() for p in model.parameters()]
                addresses = current if addresses is None else addresses
                assert current == addresses
            update()
            reference_optimizer.step()
            for parameter, expected_parameter in zip(
                model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    parameter, expected_parameter, rtol=0, atol=0
                )
                for key, state in optimizer.state[parameter].items():
                    torch.testing.assert_close(
                        state,
                        reference_optimizer.state[expected_parameter][key],
                        rtol=0,
                        atol=0,
                    )
            with torch.no_grad():
                result = _validate_loss(
                    validator, [model], [(inputs, target, {})], step=step + 1
                ).clone()
                expected_eval = (reference(inputs) - target).square().sum()
                torch.testing.assert_close(result, expected_eval, rtol=0, atol=0)
            if step == 0:
                assert validator._run_eval is not None
                assert accelerator_graph_manager._wrappers[-1]._graph is None
                assert captures == []
            elif step == 1:
                assert captures == [(1, "validation")]
            torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
        assert len(accelerator_graph_manager._wrappers) == (3 if optimizer_graph else 2)
        assert captures == [(1, "validation"), (2, "train")] + (
            [(2, "optimizer")] if optimizer_graph else []
        )
    finally:
        accelerator_graph_teardown()


def test_eval_warmup_keeps_rng_and_buffer_updates():
    torch.manual_seed(42)
    model = torch.nn.Module()
    model.register_buffer("counter", torch.zeros((), device="cuda"))
    reference_counter = torch.zeros_like(model.counter)
    inputs = torch.ones(4, 16, device="cuda")

    def forward(value):
        model.counter.add_(1)
        return value + torch.rand_like(value) + model.counter

    model.forward = forward
    validator = _validator()
    accelerator_graph_manager.capture_allowed = False
    with torch.no_grad():
        for run in range(10):
            value = inputs * (run + 1)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                reference_counter.add_(1)
                expected = (
                    (value + torch.rand_like(value) + reference_counter).square().sum()
                )
                expected_rng = torch.cuda.get_rng_state()
            actual = _validate_loss(
                validator,
                [model],
                [(value, torch.zeros_like(value), {})],
                step=run + 1,
            ).clone()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(model.counter, reference_counter, rtol=0, atol=0)
            assert torch.equal(torch.cuda.get_rng_state(), expected_rng)


def test_per_group_graph_waits_for_eval_capture():
    accelerator_graph_manager.capture_allowed = False
    parameter = torch.nn.Parameter(torch.ones(1024, device="cuda"))
    inputs = torch.ones_like(parameter)

    def body(value):
        loss = (parameter * value).sum()
        loss.backward()
        return loss.detach()

    train = wrap_with_accelerator_graph(body, name="train", num_warmup_iterations=2)
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    model.forward = lambda value: parameter * value
    validator = _validator()
    prepared = [(inputs, torch.zeros_like(inputs), {})]
    try:
        for _ in range(4):
            train(inputs)
        assert accelerator_graph_manager._wrappers[0]._graph is None
        with torch.no_grad():
            _validate_loss(validator, [model], prepared, step=1)
        for _ in range(4):
            train(inputs)
        assert accelerator_graph_manager._wrappers[0]._graph is None
        with torch.no_grad():
            _validate_loss(validator, [model], prepared, step=2)
        assert accelerator_graph_manager._wrappers[-1]._graph is not None
        for count in (1, 4, 2, 5):
            parameter.grad.zero_()
            for _ in range(count):
                train(inputs)
            torch.testing.assert_close(parameter.grad, inputs * count, rtol=0, atol=0)
            with torch.no_grad():
                assert (
                    _validate_loss(validator, [model], prepared, step=3).item() == 1024
                )
    finally:
        accelerator_graph_teardown()


def test_varlen_eval_replays_different_document_counts():
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://localhost:{get_free_port()}",
        rank=0,
        world_size=1,
    )
    torch.manual_seed(42)
    parallelism = ParallelismConfig()
    model = build_model_config(
        "debugmodel", seq_len=8, attn_backend="varlen", parallelism=parallelism
    ).build()
    model = model.cuda()
    model.init_weights(buffer_device=torch.device("cuda"))
    validator = _validator()
    validator.parallelism = parallelism
    validator.parallelism_context = ParallelismContext(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        enable_sequence_parallel=False,
    )
    validator.dl_config = SimpleNamespace(max_num_documents=4)
    validator.seq_len = 8
    validator.loss_fn = lambda output, labels: (
        torch.nn.functional.cross_entropy(output, labels, reduction="sum"),
        {},
    )
    try:
        model.eval()
        with torch.no_grad():
            for step in range(10):
                positions = (
                    [0, 1, 2, 3, 0, 1, 2, 3]
                    if step % 2 == 0
                    else [0, 1, 2, 0, 1, 2, 0, 1]
                )
                batch = {
                    "input": torch.randint(0, 2048, (8,), device="cuda"),
                    "labels": torch.randint(0, 2048, (8,), device="cuda"),
                    "positions": torch.tensor(positions, device="cuda"),
                    "padding_mask": torch.zeros(8, dtype=torch.bool, device="cuda"),
                }
                with validator.parallelism_context.activate_spmd():
                    inputs, labels, model_kwargs, _ = model.preprocess_inputs(
                        batch,
                        parallelism_context=validator.parallelism_context,
                        parallelism=validator.parallelism,
                        max_num_documents=validator.dl_config.max_num_documents,
                        max_context_length=validator.seq_len,
                    )
                prepared = [(inputs, labels, model_kwargs)]
                metadata = prepared[0][2]["attention_metadata"][VarlenInnerAttention]
                assert metadata.cu_seq_q.shape == (6,)
                assert metadata.max_q == metadata.max_k == 8
                expected = validator._validate_body([model], prepared).clone()
                actual = _validate_loss(
                    validator, [model], prepared, step=step + 1
                ).clone()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        accelerator_graph_teardown()
        dist.destroy_process_group()


def _measure_pool(mode, results):
    torch.cuda.set_device(0)
    parameter = torch.nn.Parameter(torch.ones(8 * 1024 * 1024, device="cuda"))
    inputs = torch.ones_like(parameter)
    accelerator_graph_manager.maybe_initialize()
    optimizer = torch.optim.SGD([parameter], lr=0.01)

    def eval_body():
        scratch = torch.empty_like(parameter)
        scratch.fill_(2)
        return scratch.sum()

    def train_body():
        loss = (parameter * inputs).sum()
        loss.backward()
        return loss.detach()

    evaluate = wrap_with_accelerator_graph(eval_body, name="eval")
    train = wrap_fwd_bwd_with_accelerator_graph(
        train_body, parameters=[parameter], name="train"
    )
    update = wrap_with_accelerator_graph(optimizer.step, name="optimizer")
    try:
        if mode == "train_first":
            train()
            update()
        if mode != "train":
            with torch.no_grad():
                evaluate()
        if mode not in ("eval", "train_first"):
            train()
            update()
        torch.cuda.synchronize()

        def pool_bytes():
            return sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if segment["segment_pool_id"] == accelerator_graph_manager.graph_pool
            )

        size = pool_bytes()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        address = parameter.grad.data_ptr() if parameter.grad is not None else None
        for _ in range(10):
            if mode != "train":
                assert evaluate().item() == 2 * inputs.numel()
            if mode != "eval":
                parameter.grad = None
                train()
                assert parameter.grad.data_ptr() == address
                torch.testing.assert_close(parameter.grad, inputs, rtol=0, atol=0)
                update()
            torch.cuda.synchronize()
            assert pool_bytes() == size
        results.put(size)
        print(
            f"mode={mode} pool_bytes={size} "
            f"peak_allocated={peak_allocated} peak_reserved={peak_reserved}",
            flush=True,
        )
    finally:
        accelerator_graph_teardown()


def test_eval_first_reuses_training_workspace():
    ctx = mp.get_context("spawn")
    sizes = {}
    for mode in ("train", "eval", "shared", "train_first"):
        results = ctx.Queue()
        process = ctx.Process(target=_measure_pool, args=(mode, results))
        process.start()
        process.join(timeout=120)
        assert process.exitcode == 0
        sizes[mode] = results.get(timeout=10)
    # One extra allocator block can hold the second member's scalar result.
    assert sizes["shared"] <= max(sizes["train"], sizes["eval"]) + 2 * 1024 * 1024
    assert sizes["train_first"] >= sizes["shared"] + 30 * 1024 * 1024


def _run_pp_train_eval(rank, port, schedule_name, reshard, csv_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4)
    accelerator_graph_manager.capture_allowed = False
    try:
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("pp", "dp"))
        pp_rank, dp_rank = rank // 2, rank % 2
        schedule_cls = (
            ScheduleInterleaved1F1B
            if schedule_name == "CSV"
            else get_schedule_class(schedule_name)
        )
        num_local = 1 if issubclass(schedule_cls, PipelineScheduleSingle) else 2
        stage_indices = [pp_rank + local * 2 for local in range(num_local)]
        has_first_stage = 0 in stage_indices
        has_last_stage = 2 * num_local - 1 in stage_indices
        models, stages = [], []
        for index in stage_indices:
            torch.manual_seed(42 + index)
            model = torch.nn.Sequential(
                torch.nn.Linear(8, 8, device="cuda"), torch.nn.ReLU()
            )
            fully_shard(model, mesh=mesh["dp"], reshard_after_forward=reshard)
            models.append(model)
            stages.append(
                PipelineStage(
                    model,
                    index,
                    2 * num_local,
                    torch.device("cuda", rank),
                    group=mesh["pp"].get_group(),
                )
            )
        loss_fn = lambda output, target: (output - target).square().sum()
        schedule = schedule_cls(
            stages if num_local > 1 else stages[0],
            n_microbatches=4,
            loss_fn=loss_fn,
            scale_grads=False,
        )
        if schedule_name == "CSV":
            if rank == 0:
                schedule._dump_csv(csv_path, format="compute_only")
            dist.barrier()
            schedule = _PipelineScheduleRuntime(
                stages, n_microbatches=4, loss_fn=loss_fn, scale_grads=False
            )
            schedule._load_csv(csv_path)
        parameters = [p for model in models for p in model.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=0.01, fused=True, capturable=True)
        reference_parts = []
        for index in range(2 * num_local):
            torch.manual_seed(42 + index)
            reference_parts.append(
                torch.nn.Sequential(
                    torch.nn.Linear(8, 8, device="cuda"), torch.nn.ReLU()
                )
            )
        reference = torch.nn.Sequential(*reference_parts)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.01, fused=True
        )
        validator = _validator(
            schedule=schedule, first=has_first_stage, last=has_last_stage
        )

        def body(inputs, target):
            losses = [] if has_last_stage else None
            schedule.step(
                inputs if has_first_stage else None,
                target=target if has_last_stage else None,
                losses=losses,
                return_outputs=False,
            )
            return torch.stack(losses).sum() if losses else inputs.new_zeros(())

        def prepare_for_capture(inputs, target):
            prepare_pipeline_schedule_for_cuda_graph(
                schedule,
                args=(inputs.chunk(4)[0],) if has_first_stage else (),
                kwargs={},
                target=target.chunk(4)[0] if has_last_stage else None,
                loss_kwargs={},
            )

        train = wrap_fwd_bwd_with_accelerator_graph(
            body,
            parameters=parameters,
            num_warmup_iterations=2,
            prepare_for_capture=prepare_for_capture,
            name="train",
        )
        update = wrap_with_accelerator_graph(
            optimizer.step, num_warmup_iterations=2, name="optimizer"
        )
        addresses = None
        for step in range(10):
            inputs = torch.full((8, 8), 0.1 * (step + 1), device="cuda")
            target = torch.full_like(inputs, 0.05 * (step + 1))
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_losses = []
            for x, y in zip(inputs.chunk(4), target.chunk(4)):
                loss = loss_fn(reference(x), y)
                loss.backward()
                expected_losses.append(loss.detach())
            actual_loss = train(inputs, target).clone()
            if has_last_stage:
                torch.testing.assert_close(
                    actual_loss, torch.stack(expected_losses).sum(), rtol=0, atol=0
                )
            for local, model in enumerate(models):
                for p, expected_p in zip(
                    model.parameters(),
                    reference_parts[stage_indices[local]].parameters(),
                ):
                    torch.testing.assert_close(
                        p.grad.to_local(),
                        expected_p.grad.chunk(2)[dp_rank],
                        rtol=0,
                        atol=0,
                    )
            if step >= 2:
                current = [p.grad.to_local().data_ptr() for p in parameters]
                addresses = current if addresses is None else addresses
                assert current == addresses
            update()
            reference_optimizer.step()
            prepared = [(x, y, {}) for x, y in zip(inputs.chunk(4), target.chunk(4))]
            with torch.no_grad():
                actual_eval = _validate_loss(
                    validator, models, prepared, step=step + 1
                ).clone()
                if has_last_stage:
                    expected_eval = torch.stack(
                        [
                            loss_fn(reference(x), y)
                            for x, y in zip(inputs.chunk(4), target.chunk(4))
                        ]
                    ).sum()
                    torch.testing.assert_close(
                        actual_eval, expected_eval, rtol=0, atol=0
                    )
            assert (accelerator_graph_manager._wrappers[-1]._graph is not None) == (
                step >= 1
            )
            for stage in stages:
                assert not stage._forward_chunk_states
        for local, model in enumerate(models):
            for p, expected_p in zip(
                model.parameters(),
                reference_parts[stage_indices[local]].parameters(),
            ):
                torch.testing.assert_close(
                    p.to_local(), expected_p.chunk(2)[dp_rank], rtol=0, atol=0
                )
                for key, state in optimizer.state[p].items():
                    expected_state = reference_optimizer.state[expected_p][key]
                    if expected_state.ndim:
                        expected_state = expected_state.chunk(2)[dp_rank]
                    actual = state.to_local() if hasattr(state, "to_local") else state
                    torch.testing.assert_close(actual, expected_state, rtol=0, atol=0)
    finally:
        accelerator_graph_teardown()
    # Let mp.spawn stop peers if this rank fails during communication.
    dist.destroy_process_group()


def _run_grouped_eval(rank, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        from torchtitan.components.loss import ChunkedLossWrapper
        from torchtitan.distributed.fsdp import post_validate_fsdp_for_cuda_graph

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.Linear(8, 8, device="cuda")
                self.lm_head = torch.nn.Linear(8, 8, device="cuda")

            def forward(self, inputs):
                return self.norm(inputs)

        torch.manual_seed(42)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        model, reference = Model(), Model()
        reference.load_state_dict(model.state_dict())
        losses, optimizers = [], []
        for module in (model, reference):
            fully_shard([module.norm, module.lm_head], mesh=mesh)
            fully_shard(module, mesh=mesh)
            loss = ChunkedLossWrapper.Config(num_chunks=2).build()
            loss.set_lm_head(module.lm_head)
            losses.append(loss)
            optimizers.append(
                torch.optim.AdamW(module.parameters(), lr=0.01, fused=True)
            )
        inputs = torch.ones(4, 8, device="cuda")
        labels = torch.zeros(4, dtype=torch.long, device="cuda")
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState

        def evaluate(x):
            with torch.no_grad():
                result, _ = losses[0](model(x), labels)
                post_validate_fsdp_for_cuda_graph([model])
                return result

        graph = wrap_with_accelerator_graph(
            evaluate, num_warmup_iterations=1, name="eval"
        )
        for step in range(10):
            inputs.fill_(1 + step / 10)
            training_losses = []
            for module, loss, optimizer in zip(
                (model, reference), losses, optimizers, strict=True
            ):
                module.train()
                optimizer.zero_grad(set_to_none=True)
                train_loss, _ = loss(module(inputs), labels)
                train_loss.backward()
                training_losses.append(train_loss.detach())
            torch.testing.assert_close(*training_losses, rtol=0, atol=0)
            for parameter, expected_parameter in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter.grad.to_local(),
                    expected_parameter.grad.to_local(),
                    rtol=0,
                    atol=0,
                )
            for module, optimizer in zip((model, reference), optimizers, strict=True):
                optimizer.step()
                module.eval()
            actual = graph(inputs).clone()
            with torch.no_grad():
                expected, _ = losses[1](reference(inputs), labels)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state = model.lm_head._get_fsdp_state()
            assert not state._modules_to_run_forward
            assert state._state_ctx.iter_forward_root is None
            assert state._training_state == TrainingState.IDLE
            assert reference.norm in (
                reference.lm_head._get_fsdp_state()._modules_to_run_forward
            )
    finally:
        accelerator_graph_teardown()
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_grouped_fsdp_eval_releases_partial_forward_state():
    mp.spawn(_run_grouped_eval, args=(get_free_port(),), nprocs=2, join=True)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA devices")
@pytest.mark.parametrize(
    "schedule,reshard",
    [
        ("1F1B", True),
        ("Interleaved1F1B", False),
        ("InterleavedZeroBubble", False),
        ("CSV", False),
    ],
)
def test_pp_fsdp_train_eval_shared_pool(schedule, reshard, tmp_path):
    mp.spawn(
        _run_pp_train_eval,
        args=(get_free_port(), schedule, reshard, str(tmp_path / "schedule.csv")),
        nprocs=4,
        join=True,
    )
