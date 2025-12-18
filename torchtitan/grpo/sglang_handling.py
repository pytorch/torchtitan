import os
from datetime import timedelta
from typing import Any, Optional

import requests
import torch

from torchtitan.config.job_config import JobConfig
from torchtitan.tools.logging import logger


def get_sglang_urls(job_config: JobConfig):
    logger.info(
        f"job_config.grpo.sglang_slurm_num_nodes: {job_config.grpo.sglang_slurm_num_nodes}"
    )
    if job_config.grpo.sglang_slurm_num_nodes > 0:
        # parse SLURM_JOB_NODELIST
        nodelist = (
            os.popen(f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}')
            .read()
            .split("\n")
        )
        urls = []
        nodelist = [node for node in nodelist if node != ""][
            -job_config.grpo.sglang_slurm_num_nodes :
        ]
        for node in nodelist:
            if node == "":
                continue
            for i in range(8 // job_config.grpo.sglang_tp):
                urls.append(f"{node}:{9000 + i}")
        return urls
    else:
        return job_config.grpo.sglang_urls


def wait_for_sglang(servers):
    for server in servers:
        logger.info(f"waiting on {server}")
        status_code = 503
        while status_code == 503:
            try:
                response = requests.get(f"http://{server}/health_generate")
                status_code = response.status_code
            except Exception:
                # No connection...
                status_code = 503
        logger.info(f"server {server} is ready")


def env_fix_for_distributed():
    export_env = os.environ.copy()
    os.environ.pop("LOCAL_RANK", None)
    os.environ.pop("RANK", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("GROUP_RANK", None)
    os.environ.pop("GROUP_WORLD_SIZE", None)
    os.environ.pop("LOCAL_WORLD_SIZE", None)
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("ROLE_RANK", None)
    os.environ.pop("ROLE_NAME", None)
    os.environ.pop("ROLE_WORLD_SIZE", None)
    os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ.pop("CUDA_MODULE_LOADING", None)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    for key in os.environ:
        if "TORCH_ELASTIC" in key:
            os.environ.pop(key)
            continue
        if "TORCHELASTIC" in key:
            os.environ.pop(key)
            continue
        if "TORCH_NCCL" in key:
            os.environ.pop(key)
            continue
    return export_env


def reset_env(env_dict):
    for key, val in env_dict.items():
        os.environ[key] = val


def env_fix_wrapper(func):
    def wrapper(*args, **kwargs):
        env_dict = env_fix_for_distributed()
        result = func(*args, **kwargs)
        reset_env(env_dict)
        return result

    return wrapper


def init_process_group(
    backend=None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    from torch.distributed.distributed_c10d import (
        _new_process_group_helper,
        _world,
        Backend,
        default_pg_timeout,
        PrefixStore,
        rendezvous,
    )

    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def get_hostname_url():
    sglang_slurm_num_nodes = int(os.environ.get("NUM_INFERENCE_NODES", -1))
    if sglang_slurm_num_nodes > 0:
        # parse SLURM_JOB_NODELIST
        nodelist = (
            os.popen(f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}')
            .read()
            .split("\n")
        )
        master_server = nodelist[0]
        return master_server
    elif sglang_slurm_num_nodes == 0:
        master_server = "localhost"
        return master_server
    else:
        return None


def param_to_sglang_data(name, needs_permute=False):
    out_permute = False
    # check if name is weird
    if "attention." in name and any(
        [
            ".wq" in name,
            ".wk" in name,
            ".wv" in name,
        ]
    ):
        # column parallel qkv, so...
        out_permute = (".wq" in name) or (".wk" in name)
        out_name = "model." + name.split("attention")[0] + "self_attn.qkv_proj"
        if ".bias" in name:
            out_name += ".bias"
        else:
            out_name += ".weight"
    elif "attention." in name and ".wo" in name:
        out_name = "model." + name.split("attention")[0] + "self_attn.o_proj"
        if ".bias" in name:
            out_name += ".bias"
        else:
            out_name += ".weight"
    elif "attention." in name:
        # QK Norm
        out_name = "model." + name.replace("attention", "self_attn")
        out_permute = True
    elif "moe" in name:

        out_name = "model." + name.replace("moe", "mlp").replace(
            ".w1", ".w13_weight"
        ).replace(".w3", ".w13_weight").replace(".w2", ".w2_weight").replace(
            ".router.gate", ".gate"
        )
    elif (".w1." in name) or (".w3" in name):
        out_name = "model." + name.replace("feed_forward", "mlp").replace(
            ".w1.", ".gate_up_proj."
        ).replace(".w3.", ".gate_up_proj.")
    elif "w1w3" in name:
        out_name = "model." + name.replace("feed_forward", "mlp").replace(
            "w1w3", "gate_up_proj"
        )
    elif "feed_forward" in name:
        out_name = "model." + name.replace("feed_forward", "mlp")
        mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for key, val in mapping.items():
            out_name = out_name.replace(key, val)
    elif "_norm" in name:
        out_name = "model." + name.replace("attention_norm", "input_layernorm").replace(
            "ffn_norm", "post_attention_layernorm"
        )
    elif "output.weight" == name:
        out_name = "lm_head.weight"
    else:
        # emb/out layer
        out_name = "model." + name.replace("tok_embeddings", "embed_tokens").replace(
            "output", "lm_head"
        )
    out_name = out_name.replace("._checkpoint_wrapped_module.", ".").replace(
        "._orig_mod.", "."
    )
    if not needs_permute:
        return out_name, False
    return out_name, out_permute


@env_fix_wrapper
def setup_group(hostname, sglang_port, total_group_size, local_rank):
    gloo_group = init_process_group(
        backend="gloo",
        init_method=f"tcp://{hostname}:{sglang_port}",
        world_size=total_group_size,
        group_name="gloo_group",
        rank=local_rank,
    )
    logger.info(f"SGlang GLOO process group created, local_rank: {local_rank}")
    nccl_group = init_process_group(
        backend="nccl",
        init_method=f"tcp://{hostname}:{sglang_port}",
        world_size=total_group_size,
        group_name="weight_update_group",
        rank=local_rank,
    )
    logger.info(f"SGlang NCCL process group created, local_rank: {local_rank}")
    return nccl_group, gloo_group


@env_fix_wrapper
def send_param(
    local_param,
    name,
    param_shape,
    weight_dtypes,
    tp_degree,
    dp_shard_degree,
    total_group_size,
    sglang_gloo_group,
    sglang_nccl_group,
    param_indx,
):
    if torch.distributed.get_rank() == 0:
        object_list = [
            {
                "name": name,
                "shape": param_shape,
                "local_shape": local_param.shape,
                "dtype": weight_dtypes[name],
            }
        ]
    else:
        object_list = [
            None,
        ]
    desired_dtype = (
        weight_dtypes[name]
        if isinstance(weight_dtypes[name], torch.dtype)
        else getattr(torch, weight_dtypes[name])
    )
    logger.debug(f"Attempting to send {object_list}")
    obj_indx = torch.LongTensor([param_indx]).to(device=local_param.device)
    torch.distributed.broadcast(obj_indx, group_src=0, group=sglang_nccl_group)
    # setup tensor list
    tensor_list = [
        torch.zeros(
            local_param.shape if indx < (dp_shard_degree * tp_degree) else 1,
            dtype=desired_dtype,
            device=local_param.device,
        )
        for indx in range(total_group_size)
    ]
    torch.distributed.all_gather(
        tensor_list, local_param.to(desired_dtype), group=sglang_nccl_group
    )


@env_fix_wrapper
def send_wait(sglang_nccl_group, device):
    logger.debug("Sending wait signal to sglang...")
    indx_tensor = torch.LongTensor([-1]).to(device=device)
    torch.distributed.broadcast(indx_tensor, 0, group=sglang_nccl_group)
