import math
from typing import List, Optional, Tuple

import torch
from torch.optim import Optimizer


class MarsAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.90, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        gamma=0.025,  # scaling param for gradient correction for MARS
        max_grad_norm=1.0,
    ):
        """
        Un-Official implementation of MARS AdamW optimizer in PyTorch,
        from
        "MARS: Unleashing the Power of Variance Reduction for Training Large Models" paper.
        https://arxiv.org/abs/2411.10438

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (β₁, β₂)
            eps: Term added for numerical stability
            weight_decay: Weight decay coefficient (λ)
            gamma: Scaling parameter for gradient correction (γₜ)
            max_grad_norm: Maximum norm for gradient clipping
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            gamma = group["gamma"]

            # Get parameters with gradients
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            prev_grads = []
            state_steps = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                prev_grads.append(state["prev_grad"])
                state["step"] += 1
                state_steps.append(state["step"])

            # Calculate gradient correction and clipped version
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                prev_grad = prev_grads[i]

                # Calculate cₜ (gradient with correction term)
                correction = gamma * beta1 / (1 - beta1) * (grad - prev_grad)
                c_t = grad + correction

                # Gradient clipping (if necessary)
                grad_norm = torch.norm(c_t)
                if grad_norm > group["max_grad_norm"]:
                    c_t = c_t * group["max_grad_norm"] / grad_norm

                # Update exponential moving averages
                exp_avgs[i].mul_(beta1).add_(c_t, alpha=1 - beta1)
                exp_avg_sqs[i].mul_(beta2).addcmul_(c_t, c_t, value=1 - beta2)

                # Store current gradient for next iteration
                prev_grads[i].copy_(grad)

                step = state_steps[i]

                # Weight decay (check if this needs to be before update)
                if group["weight_decay"] > 0:
                    param.data.add_(
                        param.data, alpha=-group["lr"] * group["weight_decay"]
                    )

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Calculate step size
                step_size = group["lr"] / bias_correction1

                # Calculate denominator
                denom = (exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                # Update parameters
                param.data.addcdiv_(exp_avgs[i], denom, value=-step_size)

                # Weight decay (check if this needs to be before update)
                if group["weight_decay"] > 0:
                    param.data.add_(
                        param.data, alpha=-group["lr"] * group["weight_decay"]
                    )

        return loss
