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

from torchtitan.components.validate import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.cuda_graph import (
    _manager,
    cuda_graph_teardown,
    set_cuda_graph_capture_enabled,
    wrap_fwd_bwd_with_cuda_graph,
    wrap_with_cuda_graph,
)
from torchtitan.distributed.pipeline_parallel import initialize_pipeline_schedule
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.muse_glimmer import build_model_config


@pytest.fixture(autouse=True)
def reset_cuda_graph_manager():
    cuda_graph_teardown()
    yield
    cuda_graph_teardown()


def _validator(*, schedule=None, first=True, last=True):
    validator = object.__new__(Validator)
    validator.config = SimpleNamespace(steps=1)
    validator._cuda_graph_enabled = True
    validator._run_eval = None
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
        side_effect=lambda row, **kwargs: row["prepared"],
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
    set_cuda_graph_capture_enabled(False)
    validator = _validator()

    def body(inputs, target):
        loss = (model(inputs) - target).square().sum()
        loss.backward()
        return loss.detach()

    train = wrap_fwd_bwd_with_cuda_graph(
        body,
        parameters=model.parameters(),
        num_warmup_iterations=2,
        name="train",
    )
    update = (
        wrap_with_cuda_graph(optimizer.step, num_warmup_iterations=2, name="optimizer")
        if optimizer_graph
        else optimizer.step
    )
    addresses = None
    captures = []

    def record_capture(name, boundary):
        if boundary == "after_capture":
            captures.append((step, name))

    monkeypatch.setattr(_manager, "log_memory", record_capture)
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
            for run in range(2 if step == 1 else 1):
                if step == 1 and run == 1:
                    set_cuda_graph_capture_enabled(True)
                with torch.no_grad():
                    result = _validate_loss(
                        validator, [model], [(inputs, target, {})], step=step + 1
                    ).clone()
                    expected_eval = (reference(inputs) - target).square().sum()
                    torch.testing.assert_close(result, expected_eval, rtol=0, atol=0)
            if step == 0:
                assert validator._run_eval is not None
                assert _manager._wrappers[-1]._graph is None
                assert captures == []
            elif step == 1:
                assert captures == [(1, "validation")]
            torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
        assert len(_manager._wrappers) == (3 if optimizer_graph else 2)
        assert captures == [(1, "validation"), (2, "train")] + (
            [(2, "optimizer")] if optimizer_graph else []
        )
    finally:
        cuda_graph_teardown()


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
    set_cuda_graph_capture_enabled(False)
    with torch.no_grad():
        for run in range(10):
            if run == 1:
                set_cuda_graph_capture_enabled(True)
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
    set_cuda_graph_capture_enabled(False)
    parameter = torch.nn.Parameter(torch.ones(1024, device="cuda"))
    inputs = torch.ones_like(parameter)

    def body(value):
        loss = (parameter * value).sum()
        loss.backward()
        return loss.detach()

    train = wrap_with_cuda_graph(body, name="train", num_warmup_iterations=2)
    evaluate = wrap_with_cuda_graph(
        lambda value: (parameter * value).sum(), name="eval"
    )
    try:
        for _ in range(4):
            train(inputs)
        assert _manager._wrappers[0]._graph is None
        set_cuda_graph_capture_enabled(True)
        with torch.no_grad():
            evaluate(inputs)
        for count in (1, 4, 2, 5):
            parameter.grad.zero_()
            for _ in range(count):
                train(inputs)
            torch.testing.assert_close(parameter.grad, inputs * count, rtol=0, atol=0)
            with torch.no_grad():
                assert evaluate(inputs).item() == 1024
    finally:
        cuda_graph_teardown()


def test_varlen_eval_replays_different_document_counts():
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://localhost:{get_free_port()}",
        rank=0,
        world_size=1,
    )
    torch.manual_seed(42)
    model = build_model_config("debugmodel", seq_len=8, attn_backend="varlen").build()
    model = model.cuda()
    model.init_weights(buffer_device=torch.device("cuda"))
    validator = _validator()
    validator.parallelism = ParallelismConfig()
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
                    prepared = [
                        model.preprocess_inputs(
                            batch,
                            parallelism_context=validator.parallelism_context,
                            parallelism=validator.parallelism,
                            max_num_documents=validator.dl_config.max_num_documents,
                            max_context_length=validator.seq_len,
                        )
                    ]
                metadata = prepared[0][2]["attention_metadata"][VarlenInnerAttention]
                assert metadata.cu_seq_q.shape == (6,)
                assert metadata.max_q == metadata.max_k == 8
                expected = validator._evaluate_body([model], prepared).clone()
                actual = _validate_loss(
                    validator, [model], prepared, step=step + 1
                ).clone()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        cuda_graph_teardown()
        dist.destroy_process_group()


def _measure_pool(mode, optimizer_graph, results):
    torch.cuda.set_device(0)
    parameter = torch.nn.Parameter(torch.ones(8 * 1024 * 1024, device="cuda"))
    inputs = torch.ones_like(parameter)
    _manager.maybe_initialize()
    eval_pool = torch.cuda.graph_pool_handle() if mode == "private" else None
    optimizer = torch.optim.SGD([parameter], lr=0.01)

    def eval_body():
        scratch = torch.empty_like(parameter)
        scratch.fill_(2)
        return scratch.sum()

    def train_body():
        loss = (parameter * inputs).sum()
        loss.backward()
        return loss.detach()

    evaluate = wrap_with_cuda_graph(eval_body, name="eval")
    train = wrap_fwd_bwd_with_cuda_graph(
        train_body, parameters=[parameter], name="train"
    )
    update = (
        wrap_with_cuda_graph(optimizer.step, name="optimizer")
        if optimizer_graph
        else optimizer.step
    )
    try:
        if mode == "train_first":
            train()
            update()
        if mode != "train":
            with (
                patch.object(_manager, "_graph_pool", eval_pool)
                if eval_pool is not None
                else nullcontext()
            ):
                evaluate()
        if mode not in ("eval", "train_first"):
            train()
            update()
        torch.cuda.synchronize()
        pools = {_manager.graph_pool}
        if eval_pool is not None:
            pools.add(eval_pool)

        def pool_bytes():
            return sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if segment["segment_pool_id"] in pools
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
            f"mode={mode} optimizer_graph={optimizer_graph} pool_bytes={size} "
            f"peak_allocated={peak_allocated} peak_reserved={peak_reserved}",
            flush=True,
        )
    finally:
        cuda_graph_teardown()


@pytest.mark.parametrize("optimizer_graph", [False, True])
def test_eval_first_reuses_training_workspace(optimizer_graph):
    ctx = mp.get_context("spawn")
    sizes = {}
    for mode in ("train", "eval", "shared", "private", "train_first"):
        results = ctx.Queue()
        process = ctx.Process(
            target=_measure_pool, args=(mode, optimizer_graph, results)
        )
        process.start()
        process.join(timeout=120)
        assert process.exitcode == 0
        sizes[mode] = results.get(timeout=10)
    # One extra allocator block can hold the second member's scalar result.
    assert sizes["shared"] <= max(sizes["train"], sizes["eval"]) + 2 * 1024 * 1024
    assert sizes["private"] >= sizes["shared"] + 30 * 1024 * 1024
    assert sizes["train_first"] >= sizes["shared"] + 30 * 1024 * 1024


def _run_pp_train_eval(rank, port, schedule_name, reshard, csv_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4)
    set_cuda_graph_capture_enabled(False)
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

        train = wrap_fwd_bwd_with_cuda_graph(
            body,
            parameters=parameters,
            num_warmup_iterations=2,
            name="train",
        )
        update = wrap_with_cuda_graph(
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
            initialize_pipeline_schedule(
                schedule,
                args=(inputs.chunk(4)[0],) if has_first_stage else (),
                kwargs={},
                target=target.chunk(4)[0] if has_last_stage else None,
                loss_kwargs={},
            )
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
                        actual = (
                            state.to_local() if hasattr(state, "to_local") else state
                        )
                        torch.testing.assert_close(
                            actual, expected_state, rtol=0, atol=0
                        )
            prepared = [(x, y, {}) for x, y in zip(inputs.chunk(4), target.chunk(4))]
            for run in range(2 if step == 1 else 1):
                if step == 1 and run == 1:
                    set_cuda_graph_capture_enabled(True)
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
            assert (_manager._wrappers[-1]._graph is not None) == (step >= 1)
            for stage in stages:
                assert not stage._forward_chunk_states
    finally:
        cuda_graph_teardown()
    # Let mp.spawn stop peers if this rank fails during communication.
    dist.destroy_process_group()


def _run_grouped_eval(rank, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8, device="cuda"),
            torch.nn.Linear(8, 8, device="cuda"),
        )
        fully_shard([model[0], model[1]], mesh=mesh)
        fully_shard(model, mesh=mesh)
        inputs = torch.ones(2, 8, device="cuda")
        # The loss calls the head after a root forward that skips it.

        def body(x):
            output = model[0](x)
            return output.square().sum()

        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torchtitan.distributed.fsdp import finish_fsdp_eval

        model(inputs).sum().backward()
        model.zero_grad(set_to_none=True)

        def evaluate(x):
            with torch.no_grad():
                result = body(x)
                finish_fsdp_eval([model])
                return result

        evaluate(inputs)
        graph = wrap_with_cuda_graph(evaluate, name="eval")
        for _ in range(10):
            actual = graph(inputs).clone()
            with torch.no_grad():
                expected = body(inputs)
                finish_fsdp_eval([model])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state = model[0]._get_fsdp_state()
            assert not state._modules_to_run_forward
            assert state._state_ctx.iter_forward_root is None
            assert state._training_state == TrainingState.IDLE
    finally:
        cuda_graph_teardown()
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
        ("GPipe", True),
        ("1F1B", False),
        ("Interleaved1F1B", False),
        ("LoopedBFS", False),
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
