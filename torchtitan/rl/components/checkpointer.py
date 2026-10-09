# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful


CONTROLLER_STATE_KEY = "controller"
DATALOADER_STATE_KEY = "dataloader"


class MirroredState(Stateful):
    """Checkpoint state produced elsewhere and mirrored into the trainer.

    ``value`` is replaced by the controller before saving the next trainer
    checkpoint. ``loaded`` holds the value restored by the checkpointer so the
    controller can retrieve it after actor startup. Separate fields prevent a
    fresh value from being mistaken for restored state.
    """

    def __init__(self) -> None:
        self.value: Any = None
        self.loaded: Any = None

    def state_dict(self) -> dict[str, bytes]:
        return {"state": pickle.dumps(self.value)}

    def load_state_dict(self, state_dict: dict[str, bytes]) -> None:
        self.loaded = pickle.loads(state_dict["state"])
