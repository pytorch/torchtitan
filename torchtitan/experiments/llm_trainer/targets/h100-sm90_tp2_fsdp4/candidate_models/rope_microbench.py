from __future__ import annotations

import os

import torch

from rope_extension import rope_backward_pair, rope_forward_pair


def bench(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def graph_bench(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()

    def replay():
        graph.replay()
        return outputs

    return bench(replay)


def main() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    q = torch.randn(1, 8192, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8192, 4, 128, device="cuda", dtype=torch.bfloat16)
    freqs_cis = torch.randn(8192, 64, device="cuda", dtype=torch.complex64)
    freqs = freqs_cis.view(1, 8192, 1, 64)

    grad_k = torch.randn_like(k)
    grad_q = torch.randn_like(q)

    def sep_forward():
        qf = q.float().view(1, 8192, 16, 64, 2)
        kf = k.float().view(1, 8192, 4, 64, 2)
        q_out = (
            torch.view_as_real(torch.view_as_complex(qf) * freqs)
            .view(1, 8192, 16, 128)
            .to(torch.bfloat16)
        )
        k_out = (
            torch.view_as_real(torch.view_as_complex(kf) * freqs)
            .view(1, 8192, 4, 128)
            .to(torch.bfloat16)
        )
        return q_out, k_out

    def fused_forward():
        return rope_forward_pair(q, k, freqs)

    def sep_backward():
        qf = grad_q.float().view(1, 8192, 16, 64, 2)
        kf = grad_k.float().view(1, 8192, 4, 64, 2)
        freqs_conj = freqs.conj().clone()
        k_out = (
            torch.view_as_real(torch.view_as_complex(kf) * freqs_conj)
            .view(1, 8192, 4, 128)
            .to(torch.bfloat16)
        )
        q_out = (
            torch.view_as_real(torch.view_as_complex(qf) * freqs_conj)
            .view(1, 8192, 16, 128)
            .to(torch.bfloat16)
        )
        return k_out, q_out

    def fused_backward():
        return rope_backward_pair(grad_k, grad_q, freqs)

    sep_q, sep_k = sep_forward()
    fused_q, fused_k = fused_forward()
    print("forward_equal", torch.equal(sep_q, fused_q), torch.equal(sep_k, fused_k))
    print(
        "forward_maxdiff",
        float((sep_q.float() - fused_q.float()).abs().max()),
        float((sep_k.float() - fused_k.float()).abs().max()),
    )

    sep_grad_k, sep_grad_q = sep_backward()
    fused_grad_k, fused_grad_q = fused_backward()
    print(
        "backward_equal",
        torch.equal(sep_grad_k, fused_grad_k),
        torch.equal(sep_grad_q, fused_grad_q),
    )
    print(
        "backward_maxdiff",
        float((sep_grad_k.float() - fused_grad_k.float()).abs().max()),
        float((sep_grad_q.float() - fused_grad_q.float()).abs().max()),
    )

    print("forward_ms", bench(sep_forward), bench(fused_forward))
    print("forward_graph_ms", graph_bench(sep_forward), graph_bench(fused_forward))
    print("backward_ms", bench(sep_backward), bench(fused_backward))
    print("backward_graph_ms", graph_bench(sep_backward), graph_bench(fused_backward))


if __name__ == "__main__":
    main()
