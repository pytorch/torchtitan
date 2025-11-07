import json
import os
import socket
import time
from collections import defaultdict
from datetime import timedelta
from typing import Any, Optional

import torch


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


def broadcast_object_list(
    object_list: list[Any],
    src: Optional[int] = None,
    group=None,
    device: Optional[torch.device] = None,
    group_src: Optional[int] = None,
):
    """
    Basically torch.distributed.broadcast_object_list, but forced to grab from rank 0.
    This is required because the original implementation assumes that rank 0 comes from the
    default process group, even when provided with a different process group for some unholy reason.

    So we just prevent our rank 0 from broadcasting by removing the conditionals that would cause it.
    """
    global_src = group_src

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    current_device = device
    # Serialize object_list elements to tensors on src rank.
    object_sizes_tensor = torch.empty(
        len(object_list), dtype=torch.long, device=current_device
    )

    # Broadcast object sizes
    torch.distributed.broadcast(object_sizes_tensor, src=global_src, group=group)

    object_tensor = torch.empty(  # type: ignore[call-overload]
        torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
        dtype=torch.uint8,
        device=current_device,
    )

    torch.distributed.broadcast(object_tensor, src=global_src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        obj_view = object_tensor[offset : offset + obj_size]
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        object_list[i] = torch.distributed.distributed_c10d._tensor_to_object(
            obj_view, obj_size, group
        )


def get_sglang_urls():
    sglang_slurm_num_nodes = int(os.environ.get("NUM_INFERENCE_NODES", -1))
    if sglang_slurm_num_nodes > 0:
        # parse SLURM_JOB_NODELIST
        nodelist = (
            os.popen(f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}')
            .read()
            .split("\n")
        )
        master_server = nodelist[0] + ":26756"
        master_gloo_server = nodelist[0] + ":26757"
        nodelist = [node for node in nodelist if node != ""][-sglang_slurm_num_nodes:]
        master_sglang_server = nodelist[-sglang_slurm_num_nodes] + ":26758"
        return master_server, master_gloo_server, master_sglang_server, nodelist
    elif sglang_slurm_num_nodes == 0:
        # parse SLURM_JOB_NODELIST
        master_server = "localhost" + ":26756"
        master_gloo_server = "localhost" + ":26757"
        master_sglang_server = "localhost" + ":26758"
        nodelist = ["localhost"]
        return master_server, master_gloo_server, master_sglang_server, nodelist
    else:
        return None, None, None, None


def get_hostnames():
    my_ip = socket.gethostbyname(socket.gethostname())
    my_hostname = socket.gethostname()
    with open("/etc/hosts", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2 and ((parts[0] == my_ip) or (my_hostname in parts)):
                    ip = parts[0]
                    if ip.startswith("127."):
                        continue
                    hostnames = parts[1:]  # ['hgx-1', 'dell-h100-nj4-1']
                    print(f"IP: {ip}")
                    print(f"Hostnames: {hostnames}")
                    return parts
    return None


def get_json_data():
    logdir = os.environ.get("LOGDIR", None)
    if logdir is None:
        raise ValueError("LOGDIR is not set")
    while not os.path.exists(f"{logdir}/sglang_json.json"):
        print(f"Waiting for {logdir}/sglang_json.json to be created...", flush=True)
        time.sleep(1)
    # make sure the file is finished writing
    time.sleep(1)
    with open(f"{logdir}/sglang_json.json", "r") as f:
        return json.load(f)


# permute for sliced rotary
def permute(w, n_heads):
    dim1 = w.shape[0]
    dim2 = w.shape[1]
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def get_name_conversions(param_mappings):
    name_conversions = defaultdict(list)
    for name in list(param_mappings.keys()):
        name_conversions[param_mappings[name]["sglang_name"]].append(name)
    return name_conversions


# permute for sliced rotary
def permute_1d(w, n_heads):
    dim1 = w.shape[0]
    return w.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)


def weight_updater_process(state_dict, q_heads, kv_heads, tp_rank, tp_size, gpu_id):
    NUM_SGLANG_NODES = int(os.environ.get("NUM_INFERENCE_NODES", -1))
    CUDA_VISIBLE_DEVICES = str(os.environ.get("CUDA_VISIBLE_DEVICES", -1)).split(",")
    SGLANG_UPDATE_PROC_DEBUG = int(os.environ.get("SGLANG_UPDATE_PROC_DEBUG", 0))
    # if NUM_SGLANG_NODES == -1:
    #     print(f"NUM_SGLANG_NODES is not set, exiting weight updater process", flush=True)
    #     return
    world_size = NUM_SGLANG_NODES * 8 if NUM_SGLANG_NODES != 0 else 4
    hostnames = get_hostnames()
    # process for together cluster...
    for key, val in state_dict.items():
        print(f"{key}: {val.shape}", flush=True)
    master_addr, master_gloo_addr, master_sglang_addr, urls = get_sglang_urls()
    torch.cuda.set_device(tp_rank)
    print(
        f"Beginning weight updater process on TP rank {tp_rank} of {tp_size} with q heads: {q_heads} and "
        f"kv heads: {kv_heads} and gpu_id {gpu_id} with CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}"
        f" with hostname: {hostnames[1]}, master {master_addr}, and world size: {world_size}",
        flush=True,
    )
    if master_addr is None:
        print("Master address is None, exiting weight updater process", flush=True)
        return
    rank = -1
    # Probably always 4 in single node? But just in case someone needs tp for whatever reason
    ranks_per_node = 8 if NUM_SGLANG_NODES != 0 else 4
    if NUM_SGLANG_NODES == 0:
        rank = (
            int(CUDA_VISIBLE_DEVICES[gpu_id]) - 4
        )  # Skip the first 4 ranks for single node
    else:
        for i, url in enumerate(urls):
            print(f"Worker {i} url: {url}", flush=True)
            if url in hostnames:
                rank = ranks_per_node * i + int(CUDA_VISIBLE_DEVICES[gpu_id])
    if rank == -1:
        print("Rank is -1, exiting weight updater process", flush=True)
        return
    print("Getting json...", flush=True)
    json_data = get_json_data()
    param_name_list = list(json_data["param_mappings"].keys())
    param_name_list.sort()
    if rank == 0:
        print("Rank 0, writing json of weight dtypes...", flush=True)
        name_conversions = get_name_conversions(json_data["param_mappings"])
        weight_dtypes = {}
        for name in state_dict.keys():
            tt_names = name_conversions[name]
            for tt_name in tt_names:
                weight_dtypes[tt_name] = str(state_dict[name].dtype).split(".")[-1]
        with open(f"{os.environ['LOGDIR']}/sglang_dtypes.json", "w") as f:
            json.dump(weight_dtypes, f)
    print("Got json", flush=True)
    num_training_gpus = json_data["dp_shard_degree"] * json_data["tp_degree"]
    total_group_size = num_training_gpus + world_size
    rank = rank + num_training_gpus  # scale rank up by num_training_gpus
    print(f"Total group size: {total_group_size}", flush=True)
    print(f"Num training gpus: {num_training_gpus}", flush=True)
    print(f"Creating process group with rank {rank} of {total_group_size}", flush=True)
    sglang_group = torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_sglang_addr}",
        world_size=world_size,
        rank=rank - num_training_gpus,
        group_name="sglang_group",
    )
    print("Created SGLang Process group", flush=True)
    gloo_group = init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}",
        world_size=total_group_size,
        rank=rank,
        group_name="gloo_group",
    )
    print("Created Gloo group", flush=True)
    nccl_group = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}",
        world_size=total_group_size,
        rank=rank,
        group_name="weight_update_group",
    )
    print("Created NCCL Process group", flush=True)
    print("Initialized process group", flush=True)
    my_device = list(state_dict.values())[0].device
    with torch.no_grad():
        qkv_buffer = {}
        gate_up_buffer = {}
        qkv_bias_buffer = {}
        while True:
            object_list = [
                None,
            ]
            obj_indx = torch.zeros(1, dtype=torch.long).to(device=my_device)
            torch.distributed.broadcast(obj_indx, group_src=0, group=nccl_group)
            tt_indx = obj_indx.item()
            if tt_indx == -1:
                continue
            tt_name = param_name_list[tt_indx]
            name = json_data["param_mappings"][tt_name]["sglang_name"]
            shape = json_data["param_mappings"][tt_name]["shape"]
            # dtype = json_data["param_mappings"][tt_name]["dtype"]
            local_shape = json_data["param_mappings"][tt_name]["local_shape"]
            target_dtype = state_dict[name].dtype
            if SGLANG_UPDATE_PROC_DEBUG == 1:
                print(
                    f"Received tt_indx: {tt_indx}, Orig: {state_dict[name].shape}, {state_dict[name].dtype}, "
                    f"assumed local shape: {local_shape}, target dtype: {target_dtype}",
                    flush=True,
                )
            # setup tensor list
            tensor_list = [
                torch.zeros(
                    local_shape if indx < num_training_gpus else 1,
                    dtype=target_dtype,
                    device=state_dict[name].device,
                )
                for indx in range(total_group_size)
            ]
            torch.distributed.all_gather(
                tensor_list,
                torch.zeros(1, dtype=target_dtype, device=state_dict[name].device),
                group=nccl_group,
            )
            tensor_list = tensor_list[:num_training_gpus]  # remove dummy tensors
            # Now merge them together...
            # First, data parallel...
            if json_data["dp_shard_degree"] > 1:
                tensor_parallel_tensors = []
                for i in range(json_data["tp_degree"]):
                    tensor_parallel_tensors.append(
                        torch.cat(tensor_list[i :: json_data["tp_degree"]], dim=0)
                    )
                if json_data["tp_degree"] > 1:
                    if tensor_parallel_tensors[0].shape == state_dict[name].shape:
                        tensor = tensor_parallel_tensors[0].contiguous()
                    else:
                        tensor = torch.cat(
                            tensor_parallel_tensors,
                            dim=json_data["param_mappings"][tt_name]["tp_shard_dim"],
                        ).contiguous()
                else:
                    tensor = tensor_parallel_tensors[0].contiguous()
            else:
                # No fsdp?
                tensor = torch.cat(
                    tensor_list,
                    dim=json_data["param_mappings"][tt_name]["tp_shard_dim"],
                ).contiguous()
            if tensor.dtype != state_dict[name].dtype:
                tensor = tensor.to(state_dict[name].dtype)

            if "qkv_proj.weight" in name:
                key_val = (
                    "q" if ".wq." in tt_name else "v" if ".wv." in tt_name else "k"
                )
                if key_val == "q":
                    tensor = permute(tensor, q_heads)
                elif key_val == "k":
                    tensor = permute(tensor, kv_heads)
                qkv_buffer[key_val] = tensor
                if len(qkv_buffer) == 3:
                    # cat them all together
                    tensor = torch.cat(
                        [qkv_buffer["q"], qkv_buffer["k"], qkv_buffer["v"]], dim=0
                    ).contiguous()
                    qkv_buffer = {}
                    state_dict[name].data.copy_(tensor)
            elif "gate_up_proj.weight" in name:
                key_val = "w1" if ".w1." in tt_name else "w3"
                gate_up_buffer[key_val] = tensor
                if len(gate_up_buffer) == 2:
                    # cat them all together
                    tensor = torch.cat(
                        [gate_up_buffer["w1"], gate_up_buffer["w3"]], dim=0
                    ).contiguous()
                    gate_up_buffer = {}
                    state_dict[name].data.copy_(tensor)
            elif "qkv_proj.bias" in name:
                key_val = (
                    "q" if ".wq." in tt_name else "v" if ".wv." in tt_name else "k"
                )
                if key_val == "q":
                    tensor = permute_1d(tensor, q_heads)
                elif key_val == "k":
                    tensor = permute_1d(tensor, kv_heads)
                qkv_bias_buffer[key_val] = tensor
                if len(qkv_bias_buffer) == 3:
                    # cat them all together
                    tensor = torch.cat(
                        [
                            qkv_bias_buffer["q"],
                            qkv_bias_buffer["k"],
                            qkv_bias_buffer["v"],
                        ],
                        dim=0,
                    ).contiguous()
                    qkv_bias_buffer = {}
                    state_dict[name].data.copy_(tensor)
            elif json_data["param_mappings"][tt_name]["needs_permute"]:
                if len(shape) == 2:
                    tensor = permute(tensor, shape[0]).contiguous()
                elif len(shape) == 1:
                    if "q_norm" in name or "k_norm" in name:
                        tensor = permute_1d(tensor, 1).contiguous()
                    else:
                        tensor = permute_1d(tensor, shape[0]).contiguous()
                else:
                    raise ValueError(
                        f"Tensor {name} has shape {shape} and needs permute, but is not 1D or 2D"
                    )
                state_dict[name].data.copy_(tensor)
            else:
                state_dict[name].data.copy_(tensor)
