"""
Credit: https://github.com/jiaweizzhao/GaLore/tree/master
(copied over and condensed for convenience)
"""

import math
from typing import Callable, Iterable, Tuple

import torch
import torch.nn as nn

from torchtitan.logging_utils import logger


class GaLoreAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                if "dim" not in group:
                    group["dim"] = 2

                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                            proj_type=group["proj_type"],
                        )
                    grad = state["projector"].project(grad, state["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                norm_grad = exp_avg / denom

                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class GaLoreProjector:
    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad: torch.Tensor, iter_idx: int):
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    )
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    )
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    )
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    )
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "full":
            if self.ortho_matrix is None or iter_idx % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full"
                )
            low_rank_grad = (
                torch.matmul(self.ortho_matrix[0].t(), full_rank_grad)
                @ self.ortho_matrix[1].t()
            )

        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor):
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == "reverse_std":
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == "full":
            full_rank_grad = (
                torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
            )

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, param: torch.Tensor, rank: int, type: str):
        U, s, Vh = torch.linalg.svd(param.detach(), full_matrices=False)
        if type == "right":
            B = Vh[:rank, :].to(device=param.device, dtype=param.dtype)
            return B
        elif type == "left":
            A = U[:, :rank].to(device=param.device, dtype=param.dtype)
            return A
        elif type == "full":
            A = U[:, :rank].to(device=param.device, dtype=param.dtype)
            B = Vh[:rank, :].to(device=param.device, dtype=param.dtype)
            return [A, B]
        else:
            raise ValueError(f"type should be left, right, or full but got {type}")
