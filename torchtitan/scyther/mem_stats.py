import copy
from contextlib import nullcontext
from typing import Callable, Tuple

import torch
from test_model import GPT, GPTConfig, loss_fn
from torch import nn, optim
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mem_tracker import _ModState, MemTracker


def collect_mem_stats(
    model: nn.Module,
    optimizer: optim.Optimizer,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
):

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    inp, target = inp_and_target
    mem_tracker = MemTracker()
    mem_tracker.track_external(model, optimizer)
    last_snapshot = None
    with mem_tracker as mt:
        for iter_idx in range(2):
            loss = loss_fn(model(inp), target)
            loss.backward()
            if iter_idx == 1:
                last_snapshot = mt.get_tracker_snapshot("current")
            optimizer.step()
            optimizer.zero_grad()
            if iter_idx == 0:
                mt.reset_mod_stats()
    assert last_snapshot is not None
    for mod_stats in mt.memory_tracking.values():
        if _ModState.POST_BW not in mod_stats.snapshots.keys():
            mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
                copy.deepcopy(last_snapshot)
            )
    mt.display_modulewise_snapshots(depth=6, units="MiB", tabulate=True)
    mt.display_snapshot("peak", units="MiB", tabulate=True)
    if not active_fake_mode() and torch.cuda.is_available():
        dev = torch.device(torch.cuda.current_device())
        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
        cuda_max = torch.cuda.max_memory_allocated()
        accuracy = tracker_max / (cuda_max + 1)  # +1 to avoid div by 0
        print(f"Tracker Max: {tracker_max}, CUDA Max: {cuda_max}, Accuracy: {accuracy}")
        print(accuracy >= 0.9)
    module_mem_stats = copy.deepcopy(mt.memory_tracking)
    return module_mem_stats


if __name__ == "__main__":
    use_fake_mode = False
    with FakeTensorMode() if use_fake_mode else nullcontext():
        dev = torch.device(torch.cuda.current_device())
        n_layer = 6
        vocab_size = 8192
        config = GPTConfig(
            block_size=512,
            n_layer=n_layer,
            vocab_size=vocab_size,
            dropout=0.01,
            checkpoint_activations=True,
        )
        with torch.device(dev):
            model = GPT(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(1)
        bsz, seq_len = 64, 512
        src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        inp = (src, tgt)
        collect_mem_stats(model, optimizer, inp, loss_fn)
