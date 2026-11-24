# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step assignment and file retention of KernelSpanRecorder, on CPU.

A fake observer stands in for Cuspy: the test decides which kernel records
have been "delivered" before each step() call, independently of the host-side
step() calls, the way the host runs ahead of the GPU.
"""

import os
import types

import numpy as np

from torchtitan.observability.kernel_spans import (
    _KernelSpanFiles,
    _SPAN_DTYPE,
    KernelSpanRecorder,
    STEPS_FILE,
)

MARKER_GRAPH_ID = 99
COMPUTE_STREAM = 7
COMM_STREAM = 8


class _FakeObserver:
    """Hands out the records queued with kernel()/marker() on the next drain()."""

    def __init__(self) -> None:
        self._delivered: list[tuple[int, int, int, int]] = []
        self._cuspy = types.SimpleNamespace(flush=lambda sync: None)

    def kernel(self, start_ns: int, stream: int = COMPUTE_STREAM) -> None:
        self._delivered.append((0, start_ns, start_ns + 1, stream))

    def marker(self, start_ns: int) -> None:
        self._delivered.append(
            (MARKER_GRAPH_ID << 32, start_ns, start_ns + 1, COMPUTE_STREAM)
        )

    def drain(self) -> tuple[np.ndarray, ...]:
        rows, self._delivered = self._delivered, []
        columns = list(zip(*rows)) if rows else [(), (), (), ()]
        return tuple(
            np.array(column, dtype=dtype)
            for column, dtype in zip(columns, ["<u8", "<i8", "<i8", "<u8"])
        )

    def convert_time_array(self, times: np.ndarray) -> np.ndarray:
        return times

    def close(self) -> None:
        pass


def _make_recorder(
    files_dir: str, *, steps_per_file: int = 100
) -> tuple[KernelSpanRecorder, _FakeObserver]:
    """A recorder with the GPU-only parts (observer, marker graph, instantiate
    hook) replaced by fakes."""
    observer = _FakeObserver()
    recorder = KernelSpanRecorder.__new__(KernelSpanRecorder)
    recorder._observer = observer
    recorder._marker_graph = types.SimpleNamespace(replay=lambda: None)
    recorder._marker_graph_id = MARKER_GRAPH_ID
    recorder._instantiate_hook = types.SimpleNamespace(remove=lambda: None)
    recorder._init_step_state(
        0,
        _KernelSpanFiles(
            files_dir,
            global_step=0,
            rank=0,
            num_steps_retained=1000,
            steps_per_file=steps_per_file,
            graph_nodes={},
        ),
    )
    return recorder, observer


def _read_steps(files_dir: str) -> dict[int, list[int]]:
    """step -> kernel start times, over every step file in files_dir."""
    steps = {}
    for name in sorted(os.listdir(files_dir)):
        with np.load(os.path.join(files_dir, name)) as data:
            offsets = data["step_offsets"]
            for i, step in enumerate(data["steps"].tolist()):
                steps[step] = data["start_ns"][offsets[i] : offsets[i + 1]].tolist()
    return steps


def test_steps_are_cut_at_gpu_markers_while_host_runs_ahead(tmp_path) -> None:
    recorder, observer = _make_recorder(str(tmp_path))
    # The host ends steps 1-3 before the GPU has delivered anything.
    for _ in range(3):
        recorder.step()
    # The GPU then runs steps 1-3, each ending with its marker.
    for start_ns in (10, 20):
        observer.kernel(start_ns)
    observer.marker(30)
    for start_ns in (40, 50):
        observer.kernel(start_ns)
    observer.marker(60)
    observer.kernel(70)
    observer.marker(90)
    observer.kernel(100)
    recorder.step()  # Host ends step 4: steps 1-2 are finished, 3 awaits marker 4.
    assert recorder._step_ends == [(3, 90)]
    observer.marker(110)
    observer.kernel(120)  # Starts after the last marker: dropped at close.
    recorder.close()

    assert _read_steps(str(tmp_path)) == {
        1: [10, 20],
        2: [40, 50],
        3: [70],
        4: [100],
    }


def test_late_kernel_from_another_stream_stays_in_its_step(tmp_path) -> None:
    recorder, observer = _make_recorder(str(tmp_path))
    recorder.step()
    observer.kernel(10)
    observer.marker(30)
    recorder.step()
    # A comm kernel that started before marker 1 is delivered after it.
    observer.kernel(25, stream=COMM_STREAM)
    observer.kernel(40)
    observer.marker(60)
    recorder.step()
    recorder.close()

    assert _read_steps(str(tmp_path)) == {1: [10, 25], 2: [40]}


def test_files_are_evicted_oldest_first_beyond_retention(tmp_path) -> None:
    files = _KernelSpanFiles(
        str(tmp_path),
        global_step=0,
        rank=0,
        num_steps_retained=3,
        steps_per_file=2,
        graph_nodes={},
    )
    empty = np.empty(0, dtype=_SPAN_DTYPE)
    for step in range(1, 9):
        files.add_step(step, empty)

    # Each file holds 2 steps; the newest 2 files are the fewest holding >= 3.
    assert sorted(os.listdir(tmp_path)) == [
        STEPS_FILE.format(first_step=5, last_step=6),
        STEPS_FILE.format(first_step=7, last_step=8),
    ]


def test_restart_deletes_abandoned_files_and_keeps_evicting_the_rest(
    tmp_path,
) -> None:
    for first_step in (1, 3, 5, 7):
        name = STEPS_FILE.format(first_step=first_step, last_step=first_step + 1)
        (tmp_path / name).touch()
    (tmp_path / "notes.txt").touch()

    # Resume from step 5: files 5-6 and 7-8 hold steps that will run again.
    files = _KernelSpanFiles(
        str(tmp_path),
        global_step=5,
        rank=0,
        num_steps_retained=3,
        steps_per_file=2,
        graph_nodes={},
    )
    assert sorted(os.listdir(tmp_path)) == [
        "notes.txt",
        STEPS_FILE.format(first_step=1, last_step=2),
        STEPS_FILE.format(first_step=3, last_step=4),
    ]

    empty = np.empty(0, dtype=_SPAN_DTYPE)
    files.add_step(6, empty)
    files.add_step(7, empty)
    # The old files are evicted with the new one, oldest first.
    assert sorted(os.listdir(tmp_path)) == [
        "notes.txt",
        STEPS_FILE.format(first_step=3, last_step=4),
        STEPS_FILE.format(first_step=6, last_step=7),
    ]
