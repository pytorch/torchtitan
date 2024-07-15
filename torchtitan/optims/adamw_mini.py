import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn


class AdamWMini(torch.optim.Optimizer):
    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, nn.Parameter]],
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        *,
        dim: int = 2048,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
    ):
        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.dim == int(self.dim):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not self.n_heads == int(self.n_heads):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        optim_groups = []
        count_embd = count_output = count_wq = count_wk = 0
        for param_name, param in named_parameters:
            if not param.requires_grad:
                continue
            state = {}
            state["name"] = param_name
            state["params"] = param
            if "norm" in param_name or "ln_f" in param_name:
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay
            if "embed" in param_name or "wte" in param_name or "embd" in param_name:
                count_embd += 1
            if "lm_head.weight" in param_name or "output.weight" in param_name:
                count_output += 1
            if "q_proj.weight" in param_name or "wq.weight" in param_name:
                count_wq += 1
                assert (
                    self.dim * self.dim
                ) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                state["head_numel"] = self.dim * self.dim // self.n_heads
            if "k_proj.weight" in param_name or "wk.weight" in param_name:
                count_wk += 1
                assert (
                    self.dim * self.dim
                ) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                state["head_numel"] = self.dim * self.dim // self.n_heads
            optim_groups.append(state)

        self.embd_names = {"embed", "embd", "wte", "lm_head.weight", "output.weight"}
        self.wqk_names = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if any(embd_name in name for embd_name in self.embd_names):
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["step"] = 0
                        state["v"] = torch.zeros_like(p, dtype=torch.float32)

                    grad = p.grad.to(torch.float32)
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)
                elif any(wqk_name in name for wqk_name in self.wqk_names):
                    dim = group["head_numel"]
                    if len(state) == 0:
                        m = torch.zeros_like(p, dtype=torch.float32)
                        state["m"] = m.view(-1, dim)
                        state["head"] = state["m"].size(0)
                        state["step"] = 0
                        # NOTE: We must use `zeros_like` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(state["head"])
                        state["vmean"] = torch.zeros_like(
                            state["m"][0 : state["head"], 0:1]
                        )

                    grad = p.grad.to(torch.float32)
                    head = state["head"]
                    grad = grad.view(head, dim)

                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(head, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                else:
                    if len(state) == 0:
                        dim = p.numel()

                        state["m"] = torch.zeros_like(p, dtype=torch.float32)
                        state["step"] = 0
                        # NOTE: We must use `new_zeros` for vmean to be a
                        # DTensor (not `torch.Tensor`) for DTensor parameters.
                        # state["vmean"] = torch.zeros(1, device=p.device)
                        state["vmean"] = p.new_zeros(1)
                        state["dim"] = dim

                    grad = p.grad.to(torch.float32)
                    tmp_lr = torch.sum(grad * grad)
                    tmp_lr = tmp_lr / state["dim"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * (stepsize.to(state["m"].device))
                    update.mul_(lr)
                    p.add_(-update)
