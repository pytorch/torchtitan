import importlib
from typing import Any, Callable, Optional

from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

from torchtitan.config_manager import JobConfig

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    has_torchft = True
else:
    has_torchft = False


def init_ft_manager(job: JobConfig) -> Optional["ft.Manager"]:
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

    return manager


def set_ft_state_dict_fns(manager: Optional["ft.Manager"], ckpt_manager) -> None:
    """
    Set the state dict for the given manager.
    """
    if manager is None:
        return

    def state_dict():
        ret = {}
        for k, v in ckpt_manager.staging_results().items():
            if k in {"model", "optimizer", "lr_schedulers"}:
                ret[k] = v
        return ret

    def load_state_dict(state_dict):
        assert state_dict is not None
        for k, v in state_dict.items():
            ckpt_manager.states[k].load_state_dict(v)

    manager.set_state_dict_fns(load_state_dict, state_dict)
