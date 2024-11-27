import torch
import triton
import triton.language as tl


@triton.jit
def mars_adamw_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    prev_grad_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    gamma,
    max_grad_norm,
    step,
    bias_correction1,
    bias_correction2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get offsets for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data into SRAM
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    prev_grad = tl.load(prev_grad_ptr + offsets, mask=mask)

    # Calculate gradient correction
    grad_diff = grad - prev_grad
    correction = gamma * beta1 / (1 - beta1) * grad_diff
    c_t = grad + correction

    # Gradient clipping
    c_t_norm = tl.sqrt(tl.sum(c_t * c_t))
    scale = tl.where(c_t_norm > max_grad_norm, max_grad_norm / c_t_norm, 1.0)
    c_t = c_t * scale

    # Update momentum
    exp_avg = beta1 * exp_avg + (1 - beta1) * c_t
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (c_t * c_t)

    # Store current gradient for next iteration
    tl.store(prev_grad_ptr + offsets, grad, mask=mask)

    # Bias correction
    step_size = lr / bias_correction1
    denom = (tl.sqrt(exp_avg_sq) / tl.sqrt(bias_correction2)) + eps

    # Update parameters with weight decay
    update = exp_avg / denom
    param = param - step_size * (update + weight_decay * param)

    # Store results
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class TritonMarsAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        gamma=0.025,
        max_grad_norm=1.0,
    ):
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
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)

                state["step"] += 1

                beta1, beta2 = group["betas"]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Grid configuration
                n_elements = p.numel()
                BLOCK_SIZE = 1024
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

                # Launch kernel
                mars_adamw_kernel[grid](
                    p,
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["prev_grad"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    group["gamma"],
                    group["max_grad_norm"],
                    state["step"],
                    bias_correction1,
                    bias_correction2,
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

        return loss


# optimizer = TritonMarsAdamW(model.parameters(), lr=6e-4, gamma=0.025)
