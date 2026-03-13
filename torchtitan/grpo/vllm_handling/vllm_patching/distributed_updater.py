import json
import os
import socket
import time
from collections import defaultdict
from datetime import timedelta
from typing import Any, Optional

import torch


def is_lora_param(name: str) -> bool:
    """Check if a parameter name corresponds to a LoRA weight."""
    return any(
        suffix in name
        for suffix in [
            "lora_a_stacked",
            "lora_b_stacked",
            ".lora_a",
            ".lora_b",
            "_lora_a",
            "_lora_b",
        ]
    )


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
    # Since we're never using < torch 2.8, we can just use the backend_options parameter name
    pg_options_param_name = "backend_options"
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


def new_group(parent_pg, backend, world_size, rank, group_name, timeout=None):
    """Create a subgroup reusing an existing PG's store - no new TCP rendezvous.

    Like ``torch.distributed.new_group`` but uses a custom parent PG
    instead of the default process group. Extracts the store from
    ``_world.pg_map[parent_pg]`` and passes it to ``_new_process_group_helper``,
    which wraps it in ``PrefixStore(group_name/, ...)`` for key isolation.
    """
    from torch.distributed.distributed_c10d import (
        _new_process_group_helper,
        _world,
        Backend,
        default_pg_timeout,
    )

    if timeout is None:
        timeout = default_pg_timeout
    if isinstance(backend, str):
        backend = Backend(backend)

    # Extract store from parent PG - same as torch.distributed.new_group
    _, parent_store = _world.pg_map[parent_pg]

    pg_options_param_name = "backend_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        parent_store,
        group_name=group_name,
        **{pg_options_param_name: None},
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


def reshape_lora_for_vllm(
    tensor: torch.Tensor, vllm_name: str, vllm_shape: tuple
) -> torch.Tensor:
    """
    Reshape a LoRA tensor from torchtitan format to vLLM stacked format.

    vLLM expects:
    - Attention LoRA: [1, 1, rank, dim] or [1, 1, dim, rank] (for single adapter)
    - MoE LoRA: [num_experts, rank, dim] or [num_experts, dim, rank]

    torchtitan provides:
    - Attention LoRA: [rank, dim] or [dim, rank]
    - MoE LoRA: [num_experts, rank, dim] or [num_experts, dim, rank]

    Args:
        tensor: The tensor from torchtitan
        vllm_name: The target vLLM parameter name (used to determine reshape needed)
        vllm_shape: The expected shape in vLLM state dict

    Returns:
        Reshaped tensor matching vLLM format
    """
    # Check if this is an attention LoRA (needs [1, 1, ...] prefix)
    is_attention_lora = any(
        proj in vllm_name
        for proj in [
            "qkv_proj.q_proj",
            "qkv_proj.k_proj",
            "qkv_proj.v_proj",
            "o_proj.o_proj",
        ]
    )

    if is_attention_lora:
        # Add batch dimensions [1, 1, ...] for vLLM stacked format
        if len(tensor.shape) == 2:
            # Shape is [rank, dim] or [dim, rank], need [1, 1, rank, dim] or [1, 1, dim, rank]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 3:
            # Already has one leading dim, add another
            tensor = tensor.unsqueeze(0)

    # For MoE LoRA, the shape should already match [num_experts, rank, dim]
    # or [num_experts, dim, rank] - no reshape needed

    # Verify shape matches expected vLLM shape
    if tensor.shape != tuple(vllm_shape):
        raise ValueError(
            f"LoRA tensor shape mismatch for {vllm_name}: "
            f"got {tensor.shape}, expected {tuple(vllm_shape)}"
        )

    return tensor


# permute for sliced rotary
def permute_1d(w, n_heads):
    dim1 = w.shape[0]
    return w.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)


def weight_updater_process(
    state_dict,
    q_heads,
    kv_heads,
    tp_rank,
    tp_size,
    ep_rank,
    ep_size,
    gpu_id,
    pp_rank=0,
    pp_size=1,
):
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
    torch.cuda.set_device(gpu_id)
    print(
        f"Beginning weight updater process on TP rank {tp_rank} of {tp_size}, "
        f"PP rank {pp_rank} of {pp_size}, "
        f"with q heads: {q_heads} and kv heads: {kv_heads} and gpu_id {gpu_id} "
        f"with CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
        f"with hostname: {hostnames[1]}, master {master_addr}, and world size: {world_size}",
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
        # Use a common dtype for all params - with PP, rank 0 may not have all
        # keys in its state_dict, but all params share the same dtype in practice.
        common_dtype = str(next(iter(state_dict.values())).dtype).split(".")[-1]
        weight_dtypes = {
            tt_name: common_dtype for tt_name in json_data["param_mappings"]
        }
        with open(f"{os.environ['LOGDIR']}/sglang_dtypes.json", "w") as f:
            json.dump(weight_dtypes, f)
    print("Got json", flush=True)
    train_pp_size = json_data.get("train_pp_size", 1)
    num_training_gpus_per_pp = json_data["dp_shard_degree"] * max(
        json_data["tp_degree"], json_data["ep_degree"]
    )
    vllm_global_rank = rank  # vLLM rank before any offset

    # ── Signal group ──────────────────────────────────────────────
    # All training dp_replicate_rank==0 ranks (across PP stages) +
    # all vLLM ranks.  Used for heartbeat/start-update signals only.
    signal_training_size = train_pp_size * num_training_gpus_per_pp
    signal_group_size = signal_training_size + world_size
    signal_rank = signal_training_size + vllm_global_rank
    print(
        f"Signal group: size={signal_group_size}, rank={signal_rank}, "
        f"train_pp_size={train_pp_size}",
        flush=True,
    )

    # Intra-vLLM group (unchanged)
    sglang_group = torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_sglang_addr}",
        world_size=world_size,
        rank=vllm_global_rank,
        group_name="sglang_group",
    )
    print("Created SGLang Process group", flush=True)
    gloo_group = init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}",
        world_size=signal_group_size,
        rank=signal_rank,
        group_name="gloo_group",
    )
    print("Created Gloo group", flush=True)
    signal_nccl_group = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}",
        world_size=signal_group_size,
        rank=signal_rank,
        group_name="weight_update_group",
    )
    print("Created signal NCCL group", flush=True)

    # ── Per-PP data groups ────────────────────────────────────────
    # One per training PP stage that maps to this vLLM PP stage.
    # Subgroups of the signal NCCL group - share its store via
    # PrefixStore namespacing, no extra TCP ports needed.
    assert (
        train_pp_size % pp_size == 0
    ), f"train_pp ({train_pp_size}) must be a multiple of vllm_pp ({pp_size})"
    vllm_ranks_per_pp = world_size // pp_size
    stages_per_vllm_pp = train_pp_size // pp_size
    pp_data_group_size = num_training_gpus_per_pp + vllm_ranks_per_pp
    # Interleaved layout: within each vLLM instance, GPUs are assigned
    # PP rank 0, PP rank 1, ... so global ranks interleave by pp_size.
    vllm_rank_within_pp = vllm_global_rank // pp_size
    pp_data_rank = num_training_gpus_per_pp + vllm_rank_within_pp

    pp_data_groups = {}  # train_pp_stage -> nccl_group
    for train_stage in range(train_pp_size):
        mapped_vllm_stage = train_stage // stages_per_vllm_pp
        if mapped_vllm_stage != pp_rank:
            continue
        group = new_group(
            signal_nccl_group,
            "nccl",
            pp_data_group_size,
            pp_data_rank,
            f"weight_data_pp{train_stage}",
        )
        pp_data_groups[train_stage] = group
        print(
            f"Created per-PP data group for train stage {train_stage}, "
            f"rank={pp_data_rank}/{pp_data_group_size}",
            flush=True,
        )

    # ── Build per-PP param lists + interleaved iteration order ────
    per_pp_params = defaultdict(list)
    for tt_name in param_name_list:
        pp_stage = json_data["param_mappings"][tt_name].get("train_pp_stage", 0)
        per_pp_params[pp_stage].append(tt_name)

    # Round-robin interleave across PP stages so NCCL can pipeline
    # ops across different groups (different communicators = different
    # CUDA streams = natural parallelism when issued back-to-back).
    interleaved_params = []
    max_params = max(len(v) for v in per_pp_params.values()) if per_pp_params else 0
    for i in range(max_params):
        for pp_stage in sorted(per_pp_params.keys()):
            if i < len(per_pp_params[pp_stage]):
                interleaved_params.append((per_pp_params[pp_stage][i], pp_stage))

    print(
        f"PP rank {pp_rank}/{pp_size}: joined {len(pp_data_groups)} data groups "
        f"(train stages {sorted(pp_data_groups.keys())}), "
        f"state_dict has {len(state_dict)} keys, "
        f"{len(interleaved_params)} interleaved params to iterate",
        flush=True,
    )
    # setup ep shard dim modifier
    gpus_per_ep_dp = json_data["ep_degree"] * json_data["dp_shard_degree"]
    # Dump interleaved iteration order for debugging
    if pp_rank == 0:
        import json as _json

        _logdir = os.environ.get("LOGDIR", "/tmp")
        vllm_debug = {
            "per_pp_params": {str(k): v for k, v in per_pp_params.items()},
            "interleaved_params": [
                {"name": n, "pp_stage": s} for n, s in interleaved_params
            ],
            "state_dict_keys": sorted(state_dict.keys()),
            "pp_data_groups_stages": sorted(pp_data_groups.keys()),
        }
        with open(f"{_logdir}/vllm_pp{pp_rank}_iteration_order.json", "w") as _f:
            _json.dump(vllm_debug, _f, indent=2)
    my_device = list(state_dict.values())[0].device
    with torch.no_grad():
        # Keyed by vLLM target name so interleaved PP stages don't
        # cross-contaminate (e.g. layers.0 vs layers.24 buffers).
        qkv_buffer = defaultdict(dict)
        gate_up_buffer = defaultdict(dict)
        qkv_bias_buffer = defaultdict(dict)
        w1w3_buffer = defaultdict(dict)
        while True:
            # Receive signal on the signal group.
            # -1 = wait/heartbeat, >0 = start weight update.
            signal = torch.zeros(1, dtype=torch.long).to(device=my_device)
            torch.distributed.broadcast(signal, group_src=0, group=signal_nccl_group)
            if signal.item() <= 0:
                continue

            # Weight update: iterate params in interleaved order across PP
            # stages.  Each all_gather goes to the per-PP data group for
            # that training PP stage.  Different groups = different NCCL
            # communicators = different CUDA streams = natural pipelining.
            total_interleaved = len(interleaved_params)
            if SGLANG_UPDATE_PROC_DEBUG:
                print(
                    f"[vLLM PP {pp_rank}] weight update: starting {total_interleaved} "
                    f"interleaved params",
                    flush=True,
                )
            for tt_name, train_pp_stage in interleaved_params:
                if train_pp_stage not in pp_data_groups:
                    if SGLANG_UPDATE_PROC_DEBUG:
                        print(
                            f"[vLLM PP {pp_rank}] skipping {tt_name} "
                            f"(train PP {train_pp_stage} not mine)",
                            flush=True,
                        )
                    continue  # not mapped to this vLLM PP stage

                data_group = pp_data_groups[train_pp_stage]
                name = json_data["param_mappings"][tt_name]["sglang_name"]
                shape = json_data["param_mappings"][tt_name]["shape"]
                local_shape = json_data["param_mappings"][tt_name]["local_shape"]

                owns_param = name in state_dict

                if owns_param:
                    target_dtype = state_dict[name].dtype
                else:
                    dtype_str = json_data["param_mappings"][tt_name]["dtype"]
                    target_dtype = getattr(torch, dtype_str)

                if SGLANG_UPDATE_PROC_DEBUG == 1 and owns_param:
                    print(
                        f"Updating {tt_name} -> {name} (train PP {train_pp_stage}), "
                        f"shape: {state_dict[name].shape}, "
                        f"local_shape: {local_shape}, dtype: {target_dtype}",
                        flush=True,
                    )

                # all_gather on the per-PP data group
                if SGLANG_UPDATE_PROC_DEBUG:
                    print(
                        f"[vLLM PP {pp_rank}] all_gather {tt_name} -> {name} "
                        f"(train PP {train_pp_stage}, owns={owns_param})",
                        flush=True,
                    )
                tensor_list = [
                    torch.zeros(
                        local_shape if indx < num_training_gpus_per_pp else 1,
                        dtype=target_dtype,
                        device=my_device,
                    )
                    for indx in range(pp_data_group_size)
                ]
                torch.distributed.all_gather(
                    tensor_list,
                    torch.zeros(1, dtype=target_dtype, device=my_device),
                    group=data_group,
                )
                if SGLANG_UPDATE_PROC_DEBUG:
                    print(
                        f"[vLLM PP {pp_rank}] all_gather {tt_name} -> {name} "
                        f"(train PP {train_pp_stage}, owns={owns_param})",
                        flush=True,
                    )

                if not owns_param:
                    del tensor_list
                    continue

                tensor_list = tensor_list[:num_training_gpus_per_pp]
                # Now merge them together...
                # First, data parallel...
                if json_data["dp_shard_degree"] > 1:
                    # TODO: support tp in ep case
                    if json_data["param_mappings"][tt_name]["ep_enabled"]:
                        expert_parallel_tensors = []
                        for i in range(json_data["ep_degree"]):
                            expert_parallel_tensors.append(
                                torch.cat(
                                    tensor_list[i :: json_data["ep_degree"]],
                                    dim=0
                                    if gpus_per_ep_dp < state_dict[name].shape[0]
                                    else 1,
                                )
                            )
                        if json_data["ep_degree"] > 1:
                            if (
                                expert_parallel_tensors[0].shape
                                == state_dict[name].shape
                            ):
                                tensor = expert_parallel_tensors[0].contiguous()
                            else:
                                tensor = torch.cat(
                                    expert_parallel_tensors,
                                    dim=0,
                                ).contiguous()
                        else:
                            tensor = expert_parallel_tensors[0].contiguous()
                    else:
                        tensor_parallel_tensors = []
                        for i in range(json_data["tp_degree"]):
                            tensor_parallel_tensors.append(
                                torch.cat(
                                    tensor_list[i :: json_data["tp_degree"]], dim=0
                                )
                            )
                        if json_data["tp_degree"] > 1:
                            if (
                                tensor_parallel_tensors[0].shape
                                == state_dict[name].shape
                            ):
                                tensor = tensor_parallel_tensors[0].contiguous()
                            else:
                                tensor = torch.cat(
                                    tensor_parallel_tensors,
                                    dim=json_data["param_mappings"][tt_name][
                                        "tp_shard_dim"
                                    ],
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

                def _debug_diff(name, old, new):
                    if SGLANG_UPDATE_PROC_DEBUG:
                        diff = (new.float() - old.float()).abs()
                        print(
                            f"[WEIGHT DIFF] {name}: mean={diff.mean().item():.6e}, "
                            f"std={diff.std().item():.6e}, "
                            f"old_mean={old.float().mean().item():.6e}, "
                            f"new_mean={new.float().mean().item():.6e}",
                            flush=True,
                        )
                        print(
                            f"[STRIDE COMP] {name}: new: {new.stride()}, "
                            f"old: {old.stride()}",
                            flush=True,
                        )

                # Check if this is a LoRA parameter by looking at the name
                if is_lora_param(name):
                    # Handle LoRA weights - they should NOT be permuted
                    if json_data["param_mappings"][tt_name]["needs_permute"]:
                        raise NotImplementedError(
                            f"LoRA weights do not support permutation at this "
                            f"time, but {tt_name} has needs_permute=True."
                        )

                    # Reshape LoRA tensor to match vLLM stacked format
                    vllm_shape = state_dict[name].shape
                    tensor = reshape_lora_for_vllm(tensor, name, vllm_shape)

                    _debug_diff(name, state_dict[name].data, tensor)
                    state_dict[name].data.copy_(tensor)
                elif "qkv_proj.weight" in name:
                    key_val = (
                        "q" if ".wq." in tt_name else "v" if ".wv." in tt_name else "k"
                    )
                    if (
                        key_val == "q"
                        and json_data["param_mappings"][tt_name]["needs_permute"]
                    ):
                        tensor = permute(tensor, q_heads)
                    elif (
                        key_val == "k"
                        and json_data["param_mappings"][tt_name]["needs_permute"]
                    ):
                        tensor = permute(tensor, kv_heads)
                    qkv_buffer[name][key_val] = tensor
                    if len(qkv_buffer[name]) == 3:
                        # cat them all together
                        tensor = torch.cat(
                            [
                                qkv_buffer[name]["q"],
                                qkv_buffer[name]["k"],
                                qkv_buffer[name]["v"],
                            ],
                            dim=0,
                        ).contiguous()
                        del qkv_buffer[name]
                        _debug_diff(name, state_dict[name].data, tensor)
                        state_dict[name].data.copy_(tensor)
                elif "gate_up_proj.weight" in name:
                    key_val = "w1" if ".w1." in tt_name else "w3"
                    gate_up_buffer[name][key_val] = tensor
                    if len(gate_up_buffer[name]) == 2:
                        # cat them all together
                        tensor = torch.cat(
                            [gate_up_buffer[name]["w1"], gate_up_buffer[name]["w3"]],
                            dim=0,
                        ).contiguous()
                        del gate_up_buffer[name]
                        _debug_diff(name, state_dict[name].data, tensor)
                        state_dict[name].data.copy_(tensor)
                elif "w13_weight" in name:
                    key_val = "w1" if ".w1" in tt_name else "w3"
                    w1w3_buffer[name][key_val] = tensor
                    if len(w1w3_buffer[name]) == 2:
                        tensor = torch.cat(
                            [w1w3_buffer[name]["w1"], w1w3_buffer[name]["w3"]], dim=1
                        ).contiguous()
                        del w1w3_buffer[name]
                        _debug_diff(name, state_dict[name].data, tensor)
                        state_dict[name].data.copy_(tensor)
                elif "qkv_proj.bias" in name:
                    key_val = (
                        "q" if ".wq." in tt_name else "v" if ".wv." in tt_name else "k"
                    )
                    if (
                        key_val == "q"
                        and json_data["param_mappings"][tt_name]["needs_permute"]
                    ):
                        tensor = permute_1d(tensor, q_heads)
                    elif (
                        key_val == "k"
                        and json_data["param_mappings"][tt_name]["needs_permute"]
                    ):
                        tensor = permute_1d(tensor, kv_heads)
                    qkv_bias_buffer[name][key_val] = tensor
                    if len(qkv_bias_buffer[name]) == 3:
                        # cat them all together
                        tensor = torch.cat(
                            [
                                qkv_bias_buffer[name]["q"],
                                qkv_bias_buffer[name]["k"],
                                qkv_bias_buffer[name]["v"],
                            ],
                            dim=0,
                        ).contiguous()
                        del qkv_bias_buffer[name]
                        _debug_diff(name, state_dict[name].data, tensor)
                        state_dict[name].data.copy_(tensor)
                elif json_data["param_mappings"][tt_name]["needs_permute"]:
                    if len(shape) == 2:
                        tensor = permute(tensor, shape[0]).contiguous()
                    elif len(shape) == 1:
                        if ("q_norm" in name or "k_norm" in name) and json_data[
                            "param_mappings"
                        ][tt_name]["needs_permute"]:
                            tensor = permute_1d(tensor, 1).contiguous()
                        else:
                            tensor = permute_1d(tensor, shape[0]).contiguous()
                    else:
                        raise ValueError(
                            f"Tensor {name} has shape {shape} and needs "
                            f"permute, but is not 1D or 2D"
                        )
                    _debug_diff(name, state_dict[name].data, tensor)
                    state_dict[name].data.copy_(tensor)
                else:
                    _debug_diff(name, state_dict[name].data, tensor)
                    state_dict[name].data.copy_(tensor)
