"""Single-GPU fwd+bwd timing of the PR #18 flex CSA kernel at the flash shape.

Same shape and harness as the gather branch's bench_csa_gather.py, flex only:
with the sink moved out of the kernel (no score_mod, no captured buffer), how
long does one CSA layer's attention take? The sink-token flex measured
932-935 ms/call; the gather kernel 99.6-99.7 ms/call.
"""

import torch

from torchtitan.models.deepseek_v4.attention import CompressedSparseAttention

T, H, D = 8192, 64, 512
RATIO, TOPK, WINDOW = 4, 512, 128
N_CMP = T // RATIO
IDX_H, IDX_D = 64, 128
OPTS = {
    "BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4,
    "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16,
}


def inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = torch.bfloat16

    def r(*s, std=1.0, grad=False, dtype=bf):
        return (torch.randn(*s, generator=g, device="cuda") * std).to(dtype).requires_grad_(grad)

    return dict(
        q=r(T, H, D, grad=True), swa_k=r(T, D, grad=True), cmp_k=r(N_CMP, D, grad=True),
        idx_q=r(T, IDX_H, IDX_D), idx_k=r(N_CMP, IDX_D), idx_w=r(T, IDX_H, std=0.1),
        attn_sink=r(H, std=0.5, grad=True), cot=r(T, H, D, dtype=torch.float32),
    )


def step(mod, i):
    out = mod(i["q"], i["swa_k"], i["cmp_k"], i["idx_q"], i["idx_k"], i["idx_w"], i["attn_sink"])
    (out.float() * i["cot"]).sum().backward()
    for k in ("q", "swa_k", "cmp_k", "attn_sink"):
        i[k].grad = None


if __name__ == "__main__":
    mod = CompressedSparseAttention.Config(
        block_size=32, kernel_options=dict(OPTS), window_size=WINDOW,
        compress_ratio=RATIO, softmax_scale=D**-0.5, index_topk=TOPK,
    ).build().cuda()
    i = inputs(0)
    for _ in range(3):
        step(mod, i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        step(mod, i)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 5
    print(f"shape: T={T} H={H} D={D} n_cmp={N_CMP} topk={TOPK} window={WINDOW}  ({torch.cuda.get_device_name()})")
    print(f"  flex-sink-rescale (PR #18)  fwd+bwd {ms:8.1f} ms/call   peak {torch.cuda.max_memory_allocated()/2**30:6.1f} GiB")
    print(f"  [sink-token flex: 932-935 ms/call | gather kernel: 99.6-99.7 ms/call]")
