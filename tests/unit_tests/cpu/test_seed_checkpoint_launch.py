# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan.train import _require_seed_checkpoint_launch


def _patch_world_size(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    monkeypatch.setattr(
        "torchtitan.train.torch.distributed.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "torchtitan.train.torch.distributed.get_world_size", lambda: world_size
    )


def test_seed_checkpoint_accepts_one_device_without_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _patch_world_size(monkeypatch, 1)

    _require_seed_checkpoint_launch(SimpleNamespace(checkpointer=object()))


def test_seed_checkpoint_rejects_multiple_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_world_size(monkeypatch, 4)

    with pytest.raises(ValueError, match=r"single device.*world size 4"):
        _require_seed_checkpoint_launch(SimpleNamespace(checkpointer=object()))


def test_seed_checkpoint_rejects_missing_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_world_size(monkeypatch, 1)

    with pytest.raises(ValueError, match=r"configure checkpointer"):
        _require_seed_checkpoint_launch(SimpleNamespace(checkpointer=None))


def test_seed_checkpoint_rejects_uninitialized_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "torchtitan.train.torch.distributed.is_initialized", lambda: False
    )

    with pytest.raises(ValueError, match=r"initialized single-device"):
        _require_seed_checkpoint_launch(SimpleNamespace(checkpointer=object()))
