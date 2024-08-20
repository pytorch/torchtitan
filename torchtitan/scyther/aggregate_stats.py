import json
from collections import OrderedDict
from typing import Callable, cast, List, Tuple, TypedDict

import torch
from ac_stats import collect_ac_tradeoff_stats
from mem_stats import collect_mem_stats
from runtime_stats import collect_runtime_stats
from test_model import GPT, GPTConfig, loss_fn
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mem_tracker import _MemRefType, _ModMemStats, _ModState


class ModStats(TypedDict):
    fqn: str
    # per-module params
    param_per_module: int
    # per-module grads
    grad_per_module: int
    # total accumulated gradients up to and including this module
    grad_total: int
    # per module fw activation size (excluding input and output)
    act_fw_per_module: int
    # per module bw activation size during peak_bw
    act_bw_per_module: int
    # per module activation grad size during peak_bw
    act_grad_per_module: int
    # total activation size up to but excluding the current module
    # includes input of the current module (i.e., output of previous module)
    act_total: int
    # Inputs to the module
    input_per_module: int
    # Outputs of the module
    output_per_module: int
    # Total fw run-time of the module
    fw_runtime_per_module: float
    # Total bw run-time of the module
    bw_runtime_per_module: float
    # Is this module a leaf module
    is_leaf: bool
    # Total ac run-time of the module
    ac_runtime: float
    # Total ac_memory for the module
    ac_memory: int
    # Number of piecewise-linear functions used for approximating ac tradeoff curve
    n_segments: int
    # Slopes of the of piecewise-linear functions
    slopes: List[float]
    # Intercepts of the of piecewise-linear functions
    intercepts: List[float]
    # X breakpoints of the of piecewise-linear functions
    breakpoints: List[float]
    # Original trade-off curve
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    fw_pre_order: List[str]
    bw_pre_order: List[str]
    fw_post_order: List[str]
    bw_post_order: List[str]
    modstats: List[ModStats]


def aggregate_stats(
    model: nn.Module,
    optimizer: optim.Optimizer,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
    dev: torch.device = torch.device(torch.cuda.current_device()),
    export_to_json: bool = False,
):
    mod_mem_stats = collect_mem_stats(model, optimizer, inp_and_target, loss_fn)
    (
        mod_runtime_stats,
        fw_pre_order,
        bw_pre_order,
        fw_post_order,
        bw_post_order,
    ) = collect_runtime_stats(model, optimizer, inp_and_target, loss_fn)
    mod_ac_tradeoff_stats = collect_ac_tradeoff_stats(model, inp_and_target, loss_fn)
    module_info: ModuleInfo = {
        "fw_pre_order": fw_pre_order,
        "bw_pre_order": bw_pre_order,
        "fw_post_order": fw_post_order,
        "bw_post_order": bw_post_order,
        "modstats": [],
    }

    for mod in model.modules():
        mod_mem_stat = mod_mem_stats.get(mod, None)
        if mod_mem_stat:
            mod_mem_stat = cast(_ModMemStats, mod_mem_stat)
            if tradeoff_stats := mod_ac_tradeoff_stats.get(mod_mem_stat.mod_fqn, None):
                ac_runtime = tradeoff_stats.ac_runtime
                ac_memory = tradeoff_stats.ac_memory
                n_segments = tradeoff_stats.n_segments
                slopes = tradeoff_stats.slopes
                intercepts = tradeoff_stats.intercepts
                breakpoints = tradeoff_stats.fit_breaks
                tradeoff_curve = tradeoff_stats.tradeoff_curve
                is_leaf = False
            else:
                ac_runtime = ac_memory = n_segments = 0
                slopes = intercepts = breakpoints = []
                tradeoff_curve: OrderedDict[float, float] = OrderedDict()
                is_leaf = True

            mod_stat: ModStats = {
                "fqn": mod_mem_stat.mod_fqn,
                "param_per_module": mod_mem_stat.parameter_mem,
                "grad_per_module": mod_mem_stat.parameter_mem,
                "grad_total": mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                    _MemRefType.GRAD
                ],
                "act_fw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.snapshots[_ModState.PRE_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.output_mem,
                ),
                "act_bw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.ACT],
                ),
                "act_grad_per_module": (
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP]
                    - mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                        _MemRefType.TEMP
                    ]
                ),
                "act_total": mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][
                    _MemRefType.ACT
                ],
                "input_per_module": mod_mem_stat.input_mem,
                "output_per_module": mod_mem_stat.output_mem,
                "fw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["fw"],
                "bw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["bw"],
                "is_leaf": is_leaf,
                "ac_runtime": ac_runtime,
                "ac_memory": ac_memory,
                "n_segments": n_segments,
                "slopes": slopes,
                "intercepts": intercepts,
                "breakpoints": breakpoints,
                "tradeoff_curve": tradeoff_curve,
            }
            module_info["modstats"].append(mod_stat)
    if export_to_json:
        with open(f"{type(model).__name__}_modules_info.json", "w") as f:
            json.dump(module_info, f, indent=2)


if __name__ == "__main__":
    with FakeTensorMode():
        dev = torch.device(torch.cuda.current_device())
        n_layer = 6
        vocab_size = 8192
        config = GPTConfig(
            block_size=512,
            n_layer=n_layer,
            n_embd=768 * 4,
            vocab_size=vocab_size,
            dropout=0.01,
            checkpoint_activations=False,
        )
        with torch.device(dev):
            model = GPT(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(1)
        bsz, seq_len = 16, 512
        src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        inp = (src, tgt)
        aggregate_stats(model, optimizer, inp, loss_fn, dev=dev, export_to_json=True)
