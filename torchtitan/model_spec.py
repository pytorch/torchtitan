from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Tuple, Type

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule


@dataclass
class BaseModelArgs:
    _enforced: str = "This field is used to enforce all fields have defaults."


class ModelProtocol(Protocol):
    def from_model_args(self, args: BaseModelArgs) -> nn.Module: ...


@dataclass
class ModelSpec:
    name: str
    cls: Type[nn.Module]
    config: Dict[str, BaseModelArgs]
    # As for now, this is a string. So it will have to be built-in to the
    # TorchTitan library. In the future, we can make this a defined class
    # that can be extended like ModelSpec.
    tokenizer: str
    parallelize_fn: Callable[[nn.Module], None]
    pipelining_fn: Callable[[nn.Module], Tuple[_PipelineSchedule, List[nn.Module]]]
