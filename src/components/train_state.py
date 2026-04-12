from typing import Any

from torch.distributed.checkpoint.stateful import Stateful


class TrainState(Stateful):
    def __init__(self):
        self.step = 0
        self.ntokens_seen = 0

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
