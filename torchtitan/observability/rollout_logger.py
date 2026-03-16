# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections.abc import Callable


class RolloutLogger:
    """Logs rollout data as JSONL for offline analysis.

    Takes any list[dict] and writes one JSON line per record. An optional
    ``filter_fn`` selects which records to keep (e.g. top/bottom by reward).

    Args:
        output_dir: Directory for rollout files.
        filename: Name of the JSONL file (default: rollouts.jsonl).
        filter_fn: Optional default filter applied to every log() call.
    """

    def __init__(
        self,
        output_dir: str,
        filename: str = "rollouts.jsonl",
        filter_fn: Callable[[list[dict]], list[dict]] | None = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self._filepath = os.path.join(output_dir, filename)
        self._file = open(self._filepath, "a")  # kept open for lifetime
        self._filter_fn = filter_fn

    def log(
        self,
        records: list[dict],
        metadata: dict | None = None,
        filter_fn: Callable[[list[dict]], list[dict]] | None = None,
    ) -> None:
        """Write rollout dicts as JSON lines.

        Args:
            records: List of rollout dicts. No schema enforced.
            metadata: Stored under ``__metadata__`` in each record to avoid
                key collisions (e.g. ``{"step": 1}``).
            filter_fn: Override the default filter for this call.
        """
        if not records:
            return
        fn = filter_fn if filter_fn is not None else self._filter_fn
        if fn is not None:
            records = fn(records)
        extra = {"__metadata__": metadata} if metadata else {}
        self._file.write(
            "\n".join(json.dumps({**r, **extra}) for r in records) + "\n"
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
