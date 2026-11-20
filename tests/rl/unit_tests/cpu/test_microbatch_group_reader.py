# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each trainer DP replica's leader packs its column of the step plan and broadcasts it to the
replica's other ranks (`MicrobatchGroupReader`): every rank must train on exactly its column's
microbatches packed in plan order from the full groups."""

from __future__ import annotations

import contextvars
import dataclasses
import threading
import time
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.rollout.finished_group import FinishedRolloutGroup
from torchtitan.rl.rollout.packed_group import pack_rollout_group
from torchtitan.rl.rollout.types import RolloutGroup, RolloutTurnID
from torchtitan.rl.trainer.batcher import Batcher, MicrobatchPacker, TrainingMicrobatch
from torchtitan.rl.trainer.microbatch_group_reader import (
    _NUM_MICROBATCH_GROUPS_READ_AHEAD,
    MicrobatchGroupReader,
    ReplicaColumn,
)
from torchtitan.rl.trainer.trainer import build_dp_replica_group
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleBuilder,
)

from tests.rl.unit_tests.cpu.test_training_sample_spec import _NAN, _rollout

_SEQ_LEN = 16
_NUM_TOKENS_PER_MICROBATCH = 32


def _groups() -> list[RolloutGroup]:
    """4 text groups of 4 rollouts: varied lengths, NaN logprobs, a two-turn sample and a
    history edit (two samples of one rollout)."""
    groups = []
    for group_id in range(4):
        rollouts = [
            _rollout(
                0,
                [([1, 2], [-0.1] * (3 + group_id))],
                group_id=group_id,
                reward=0.0,
            ),
            _rollout(
                1,
                [([1], [_NAN, -0.2, -0.3])],
                group_id=group_id,
                reward=1.0,
            ),
            _rollout(
                2,
                [([1, 2, 3], [-0.1, -0.2]), ([1, 2, 3, 200, 201, 9], [-0.1, _NAN])],
                group_id=group_id,
                reward=2.0,
            ),
            _rollout(
                3,
                [([1, 2], [-0.1]), ([9, 9, 9], [-0.1, -0.2, -0.3])],
                group_id=group_id,
                reward=3.0,
            ),
        ]
        groups.append(RolloutGroup(group_id=group_id, rollouts=rollouts))
    return groups


def _packer() -> MicrobatchPacker:
    return MicrobatchPacker.Config(per_sample_pad_multiple=1).build(
        num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
        max_context_length=_SEQ_LEN,
        pad_id=0,
        temperature=1.0,
        vision_processor=None,
    )


def _plan_columns(*, dp_degree: int, num_pp_microbatches: int) -> list[ReplicaColumn]:
    """Plan `_groups()` with the batcher, and cut each DP rank's column as the controller does."""
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
        max_context_length=_SEQ_LEN,
        num_prompts_per_train_step=4,
        dp_degree=dp_degree,
        num_pp_microbatches=num_pp_microbatches,
    )
    finished_by_id = {
        group.group_id: FinishedRolloutGroup.from_group(group) for group in _groups()
    }
    # The 4th group completes the step.
    [*_, (batch, _)] = [
        batcher.add_group(summary=finished.summary)
        for finished in finished_by_id.values()
    ]
    assert batch is not None
    controller = SimpleNamespace(
        trainer_dp_degree=dp_degree,
        _packed_groups_by_id={
            group_id: finished.packed for group_id, finished in finished_by_id.items()
        },
    )
    return Controller._replica_columns(controller, batch)


def _reference_packed_column(column: ReplicaColumn) -> list[list[TrainingMicrobatch]]:
    """The column packed in plan order from the full groups: each spec's sample materialized
    from its full group, then each microbatch packed."""
    builder = TrainingSampleBuilder.Config().build()
    full_groups = {group.group_id: group for group in _groups()}
    samples = iter(
        builder.materialize(
            training_sample_specs=[spec],
            rollout_group=full_groups[spec.rollout_id.group_id],
        )[0]
        for group in column.microbatch_groups
        for specs in group
        for spec in specs
    )
    packer = _packer()
    return [
        [packer.pack([next(samples) for _ in specs]) for specs in group]
        for group in column.microbatch_groups
    ]


def _assert_microbatches_equal(
    actual: list[list[TrainingMicrobatch]], expected: list[list[TrainingMicrobatch]]
) -> None:
    assert [len(group) for group in actual] == [len(group) for group in expected]
    for actual_group, expected_group in zip(actual, expected, strict=True):
        for actual_microbatch, expected_microbatch in zip(
            actual_group, expected_group, strict=True
        ):
            torch.testing.assert_close(
                dataclasses.asdict(actual_microbatch),
                dataclasses.asdict(expected_microbatch),
                rtol=0,
                atol=0,
                equal_nan=True,
            )


def _run_rank(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
    layout: dict[str, int],
    num_pp_microbatches: int,
    break_leader_column: bool,
) -> None:
    """One trainer rank: join its DP replica's gloo group, read its microbatch groups, and save
    them (or the error it raised) to `result_dir`."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=layout["dp_degree"],
            cp=layout["cp_degree"],
            tp=layout["tp_degree"],
            pp=layout["pp_degree"],
            ep=1,
            world_size=world_size,
            enable_sequence_parallel=False,
        )
        # The trainer's own grouping and leader choice.
        replica_group, is_leader = build_dp_replica_group(
            parallelism_context, timeout=timedelta(seconds=60)
        )
        dp_rank = parallelism_context.get_optional_mesh(
            "dp", include_singleton_axes=True
        ).get_local_rank()
        columns = _plan_columns(
            dp_degree=layout["dp_degree"], num_pp_microbatches=num_pp_microbatches
        )
        column = columns[dp_rank] if is_leader else None
        if column is not None and break_leader_column:
            # Without its rollouts, materialize raises on the leader's first group.
            column = dataclasses.replace(column, rollout_groups=[])
        reader = MicrobatchGroupReader(
            num_microbatch_groups=len(columns[dp_rank].microbatch_groups),
            replica_group=replica_group,
            column=column,
            training_sample_builder=TrainingSampleBuilder.Config().build(),
            microbatch_packer=_packer(),
        )
        try:
            microbatch_groups = list(reader)
        except Exception as error:
            Path(result_dir, f"{rank}.error").write_text(
                f"{type(error).__name__}: {error}"
            )
        else:
            torch.save(
                (microbatch_groups, reader.pack_s), Path(result_dir, f"{rank}.pt")
            )
    finally:
        dist.destroy_process_group()


def _spawn_ranks(
    tmp_path: Path,
    *,
    layout: dict[str, int],
    num_pp_microbatches: int = 1,
    break_leader_column: bool = False,
) -> int:
    world_size = (
        layout["pp_degree"]
        * layout["dp_degree"]
        * layout["cp_degree"]
        * layout["tp_degree"]
    )
    mp.spawn(
        _run_rank,
        args=(
            world_size,
            str(tmp_path / "init"),
            str(tmp_path),
            layout,
            num_pp_microbatches,
            break_leader_column,
        ),
        nprocs=world_size,
        join=True,
    )
    return world_size


@pytest.mark.parametrize(
    ("layout", "num_pp_microbatches", "ranks_by_dp_rank"),
    [
        (dict(pp_degree=1, dp_degree=2, cp_degree=1, tp_degree=2), 1, [[0, 1], [2, 3]]),
        (dict(pp_degree=2, dp_degree=1, cp_degree=1, tp_degree=2), 2, [[0, 1, 2, 3]]),
    ],
    ids=["dp2_tp2", "pp2_tp2"],
)
def test_every_rank_trains_on_the_microbatches_the_batcher_packed(
    tmp_path: Path,
    layout: dict[str, int],
    num_pp_microbatches: int,
    ranks_by_dp_rank: list[list[int]],
) -> None:
    """Every rank of a DP replica (the trainer's (pp, dp, cp, tp) layout) trains on its
    column packed in plan order from the full groups, and reports its leader's packing
    time."""
    world_size = _spawn_ranks(
        tmp_path, layout=layout, num_pp_microbatches=num_pp_microbatches
    )

    columns = _plan_columns(
        dp_degree=layout["dp_degree"], num_pp_microbatches=num_pp_microbatches
    )
    for dp_rank, ranks in enumerate(ranks_by_dp_rank):
        expected = _reference_packed_column(columns[dp_rank])
        assert all(len(group) == num_pp_microbatches for group in expected)
        pack_s_by_rank = {}
        for rank in ranks:
            microbatch_groups, pack_s_by_rank[rank] = torch.load(
                tmp_path / f"{rank}.pt", weights_only=False
            )
            _assert_microbatches_equal(microbatch_groups, expected)
        leader_pack_s = pack_s_by_rank[ranks[0]]
        assert leader_pack_s > 0
        assert set(pack_s_by_rank.values()) == {leader_pack_s}
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == sorted(
        f"{rank}.pt" for rank in range(world_size)
    )


def test_a_leader_that_fails_to_pack_fails_every_rank_of_its_replica(
    tmp_path: Path,
) -> None:
    _spawn_ranks(
        tmp_path,
        layout=dict(pp_degree=1, dp_degree=2, cp_degree=1, tp_degree=2),
        break_leader_column=True,
    )

    for rank in range(4):
        error = (tmp_path / f"{rank}.error").read_text()
        if rank in (0, 2):  # the leaders
            assert error.startswith("KeyError")
        else:
            assert error.startswith("RuntimeError: The DP replica's leader failed")
            assert "KeyError" in error


def fake_column(microbatch_groups: list[list[list[int]]]) -> ReplicaColumn:
    """A column whose specs `fake_training_sample_builder` turns into the given ints, all from
    one empty rollout group."""
    return ReplicaColumn(
        microbatch_groups=[
            [
                [
                    SimpleNamespace(
                        rollout_id=RolloutTurnID(
                            group_id=0, rollout_id=index, turn_id=0
                        )
                    )
                    for index in microbatch
                ]
                for microbatch in group
            ]
            for group in microbatch_groups
        ],
        rollout_groups=[pack_rollout_group(RolloutGroup(group_id=0, rollouts=[]))[1]],
    )


fake_training_sample_builder = SimpleNamespace(
    materialize=lambda *, training_sample_specs, rollout_group: [
        spec.rollout_id.rollout_id for spec in training_sample_specs
    ]
)


def _counting_reader(
    num_groups: int, packed: list[int], *, pack=None
) -> MicrobatchGroupReader:
    """A reader on a replica of one rank whose packer records each group it packs."""

    def default_pack(samples: list[int]) -> SimpleNamespace:
        [index] = samples
        packed.append(index)
        return SimpleNamespace(preprocess_images=lambda: index)

    return MicrobatchGroupReader(
        num_microbatch_groups=num_groups,
        replica_group=None,
        column=fake_column([[[index]] for index in range(num_groups)]),
        training_sample_builder=fake_training_sample_builder,
        microbatch_packer=SimpleNamespace(
            pack_planned=lambda specs, samples: (pack or default_pack)(samples)
        ),
    )


def test_reader_yields_in_order_and_packs_two_groups_ahead() -> None:
    packed: list[int] = []
    reader = _counting_reader(6, packed)

    assert len(reader) == 6
    for [index] in reader:
        # Wait for the expected group, then 20 ms more, so a group packed too far ahead shows.
        expected = min(index + _NUM_MICROBATCH_GROUPS_READ_AHEAD, 5)
        deadline = time.monotonic() + 5
        while max(packed) < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.02)
        assert max(packed) == expected
    assert packed == list(range(6))


def test_reader_thread_sees_the_callers_context() -> None:
    step = contextvars.ContextVar("step")
    step.set(7)
    threads = []

    def pack(samples: list[int]) -> SimpleNamespace:
        threads.append(threading.current_thread())
        return SimpleNamespace(preprocess_images=lambda: step.get())

    assert list(_counting_reader(1, [], pack=pack)) == [[7]]
    assert threads[0] is not threading.main_thread()


def test_reader_skips_queued_groups_when_training_stops() -> None:
    packed: list[int] = []
    group_1_started, group_1_done = threading.Event(), threading.Event()

    def pack(samples: list[int]) -> SimpleNamespace:
        [index] = samples
        if index == 1:
            group_1_started.set()
            time.sleep(0.2)
            group_1_done.set()
        packed.append(index)
        return SimpleNamespace(preprocess_images=lambda: index)

    iterator = iter(_counting_reader(3, packed, pack=pack))

    assert next(iterator) == [0]
    assert group_1_started.wait(timeout=5)
    # Group 1 is packing and group 2 is queued: closing cancels group 2.
    iterator.close()
    assert group_1_done.wait(timeout=5)
    time.sleep(0.05)
    assert packed == [0, 1]


def test_the_leader_frees_each_groups_bytes_before_the_next_materializes() -> None:
    """The column's bytes and their samples (copies) are never all alive at once on the
    leader: each packed rollout group's buffer goes as soon as its samples exist."""
    buffer_refs: list[weakref.ref] = []
    num_alive_at_materialize = []

    class _CountingBuilder(TrainingSampleBuilder):
        def materialize(self, *, training_sample_specs, rollout_group):
            num_alive_at_materialize.append(
                sum(buffer_ref() is not None for buffer_ref in buffer_refs)
            )
            return super().materialize(
                training_sample_specs=training_sample_specs,
                rollout_group=rollout_group,
            )

    builder = _CountingBuilder(TrainingSampleBuilder.Config())
    rollout_groups, specs = [], []
    for group in _groups()[:3]:
        summary, packed = pack_rollout_group(group)
        # The regions are views into the buffer `pack_rollout_group` filled.
        buffer_refs.append(weakref.ref(packed.rollout_regions[0].base))
        specs += builder.decide(
            rollout_group=summary.group,
            turn_lengths=summary.turn_lengths,
            num_mtp_layers=0,
        ).training_sample_specs
        rollout_groups.append(packed)
    del group, summary, packed
    reader = MicrobatchGroupReader(
        num_microbatch_groups=1,
        replica_group=None,
        column=ReplicaColumn(
            microbatch_groups=[[specs]], rollout_groups=rollout_groups
        ),
        training_sample_builder=builder,
        microbatch_packer=SimpleNamespace(
            pack_planned=lambda specs, samples: SimpleNamespace(
                preprocess_images=lambda: len(samples)
            )
        ),
    )
    del rollout_groups

    assert list(reader) == [[len(specs)]]
    assert num_alive_at_materialize == [3, 2, 1]


class _MaskedSampleBuilder(TrainingSampleBuilder):
    """Builds samples that mask their last completion token, which the specs count."""

    def _build_training_sample(self, spec, rollout, **kwargs) -> TrainingSample:
        sample = super()._build_training_sample(spec, rollout, **kwargs)
        loss_mask = sample.loss_mask.copy()
        loss_mask[-1] = False
        return dataclasses.replace(sample, loss_mask=loss_mask)


class _LongerSampleBuilder(TrainingSampleBuilder):
    """Builds samples one masked token longer than their specs."""

    def _build_training_sample(self, spec, rollout, **kwargs) -> TrainingSample:
        sample = super()._build_training_sample(spec, rollout, **kwargs)
        return dataclasses.replace(
            sample,
            token_ids=np.append(sample.token_ids, 0),
            loss_mask=np.append(sample.loss_mask, False),
            logprobs=np.append(sample.logprobs, np.float32(0)),
            advantage=np.append(sample.advantage, np.float32(0)),
        )


@pytest.mark.parametrize(
    "builder_class",
    [_MaskedSampleBuilder, _LongerSampleBuilder],
    ids=["mask", "length"],
)
def test_the_leader_raises_when_a_sample_disagrees_with_its_spec(builder_class) -> None:
    """The leader checks each sample against its spec: the trainer divides the loss by the
    specs' counts summed, so a sample whose loss mask drifts would silently rescale the loss,
    and a longer one overfills the microbatch the specs planned (here, exactly full)."""
    builder = builder_class(TrainingSampleBuilder.Config())
    summary, packed = pack_rollout_group(_groups()[0])
    [spec, *_] = builder.decide(
        rollout_group=summary.group,
        turn_lengths=summary.turn_lengths,
        num_mtp_layers=0,
    ).training_sample_specs
    reader = MicrobatchGroupReader(
        num_microbatch_groups=1,
        replica_group=None,
        column=ReplicaColumn(microbatch_groups=[[[spec]]], rollout_groups=[packed]),
        training_sample_builder=builder,
        # The spec's labels fill the microbatch.
        microbatch_packer=MicrobatchPacker.Config().build(
            num_tokens_per_microbatch_per_dp_rank=spec.num_tokens - 1,
            max_context_length=spec.num_tokens - 1,
            pad_id=0,
            temperature=1.0,
            vision_processor=None,
        ),
    )

    with pytest.raises(ValueError, match="rollout_to_training_sample_specs"):
        list(reader)
