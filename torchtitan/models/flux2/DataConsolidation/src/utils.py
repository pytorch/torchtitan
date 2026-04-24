
# This file contains utility files for loading a model, 
# as well as some basic operations / functions used in the models
import os
import importlib
from collections.abc import Mapping, Sequence 

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig


def get_obj_from_str(string, reload=False):
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py#L88-L93
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config, recursive=True, debug=False):
    """
    Instantiates an object from a config dict of form:
    {
        "target": "module.submodule.ClassOrObject",
        "params": { ... optional parameters ... }
    }

    - If the target resolves to a non-callable (e.g., enum member), it is returned as-is.
    - If it is callable and params are provided (even empty), the object is instantiated with those parameters.
    - Nested configs under 'params' are recursively instantiated if they contain a 'target' key.
    """
    # https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py#L78-L85
    if config is None:
        return None
    if isinstance(config, str) and config in ['__is_first_stage__', '__is_unconditional__']:
        return None
    if isinstance(config, torch.nn.Module):
        return config
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_object(config)
    if not isinstance(config, Mapping) or not "target" in config:
        raise KeyError(f"Expected key `target` to instantiate in ({type(config)}) {config}.")

    try:
        obj = get_obj_from_str(config["target"])
    except Exception as e:
        print(f"\n Got this error when trying to follow the target \"{config['target']}\" in\n{config}")
        raise e

    has_params = not config.get("params", None) is None
    params = config.get("params", dict())

    if isinstance(params, str) and hasattr(obj, params):
        return getattr(obj, params) # for enums or class attributes
    elif not callable(obj) or not has_params:
        return obj
    
    def recursive_instantiate(x):
        """Recursively instantiate nested configs."""
        if isinstance(x, Mapping):
            if "target" in x:
                return instantiate_from_config(x, recursive=recursive, debug=debug)
            else:
                return {k: recursive_instantiate(v) for k, v in x.items()}
        elif isinstance(x, Sequence) and not isinstance(x, str):
            return [recursive_instantiate(v) for v in x]
        else:
            return x
    
    init_params = recursive_instantiate(params)

    if debug:
        print(f"Final config for {obj}:\n{init_params}")

    return obj(**init_params)



def get_global_rank():
    for key in (
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
    ):
        if key in os.environ:
            return int(os.environ[key])
    return 0  # default: single process

def rank_zero_print(*args, **kwargs):
    if get_global_rank() == 0:
        print(*args, **kwargs)