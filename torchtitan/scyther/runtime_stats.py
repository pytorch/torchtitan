import copy
from typing import Callable, Tuple

import torch
from .runtime_estimator import RuntimeEstimator
from .test_model import GPT, GPTConfig, loss_fn
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.benchmark import timer


def collect_runtime_stats(
    model: nn.Module,
    optimizer: optim.Optimizer,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
    estimate_mode_type: str = "operator-level-cost-model",
):
    # We just need one actual iteration for estimation
    warm_up_iters, actual_iters = 1, 1
    inp, target = inp_and_target

    def inner(num_iters: int):
        for _ in range(num_iters):
            loss = loss_fn(model(inp), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Initializing optimizer states and warm-up
    inner(warm_up_iters)

    estimate_mode = RuntimeEstimator()
    with estimate_mode(estimate_mode_type=estimate_mode_type):
        start = timer()
        inner(actual_iters)
        end = timer()
    # We use only one iteration for estimation
    print(f"{estimate_mode_type=} estimation process total_time: {end-start:.3f} ms")
    estimate_mode.display_modulewise_stats(depth=4)
    return (
        copy.deepcopy(estimate_mode.mod_runtimes),
        copy.deepcopy(estimate_mode.mod_fw_pre_order),
        copy.deepcopy(estimate_mode.mod_bw_pre_order),
        copy.deepcopy(estimate_mode.mod_fw_post_order),
        copy.deepcopy(estimate_mode.mod_bw_post_order),
    )


if __name__ == "__main__":
    with FakeTensorMode():
        dev = torch.device(torch.cuda.current_device())
        n_layer = 6
        vocab_size = 8192
        config = GPTConfig(
            block_size=512,
            n_layer=n_layer,
            dropout=0.01,
            vocab_size=vocab_size,
            checkpoint_activations=False,
        )
        with torch.device(dev):
            model = GPT(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(1)
        bsz, seq_len = 64, 512
        src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        inp = (src, tgt)
        collect_runtime_stats(model, optimizer, inp, loss_fn)
