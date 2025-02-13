import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

from torchtitan.config_manager import JobConfig

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    has_torchft = True
else:
    has_torchft = False


@dataclass
class FTManager:
    manager: ft.Manager
    replicate_group_size: int


def init_ft_manager(job: JobConfig) -> Optional[FTManager]:
    """
    Initialize the FT manager for the given job.
    """
    if not job.experimental.enable_torchft:
        return None

    if not has_torchft:
        raise ImportError("torchft is not installed. Please install it.")

    pg = ft.ProcessGroupBabyNCCL()
    manager = ft.Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=None,
        state_dict=None,
        use_async_quorum=True,
        replica_id=f"torchtitan_ft_{job.experimental.ft_replica_group_id}",
    )

    return FTManager(manager, job.experimental.ft_replica_group_size)
