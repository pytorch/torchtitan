from __future__ import annotations

import statistics

import torch
from attn_gym.linear import recurrent_kda_decode
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)

SHAPES = (
    (1, 8, 128, 128),
    (8, 8, 128, 128),
    (32, 8, 128, 128),
    (128, 8, 128, 128),
    (32, 2, 128, 128),
    (32, 16, 128, 128),
    (32, 32, 128, 128),
    (32, 8, 64, 64),
    (32, 8, 256, 256),
    (32, 8, 80, 48),
)
REPEATS = 3


def strided_state(
    shape: tuple[int, ...], seed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = shape[0]
    elements = 1
    for size in shape[1:]:
        elements *= size
    storage = torch.empty(slots, elements + 29, device="cuda", dtype=torch.float32)
    state = storage[:, :elements].view(shape)
    state.copy_(seed)
    return storage, state


def event_time_us(
    fn, *, warmup: int = 20, iterations: int = 200, rounds: int = 5
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(samples)


def capture(fn) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def graph_time_us(graph: torch.cuda.CUDAGraph) -> float:
    return event_time_us(graph.replay, warmup=20, iterations=1000, rounds=5)


def benchmark_shape(
    batch: int, heads: int, key_dim: int, value_dim: int
) -> tuple[float, ...]:
    torch.manual_seed(1000 + batch + heads + key_dim + value_dim)
    dtype = torch.bfloat16
    lower_bound = -5.0
    slots = batch + 1

    per_head = torch.randn(
        batch,
        heads,
        2 * key_dim + value_dim,
        device="cuda",
        dtype=dtype,
    )
    q = per_head[..., :key_dim]
    k = per_head[..., key_dim : 2 * key_dim]
    v = per_head[..., 2 * key_dim :]
    attention_gym_qkv = per_head.flatten(1)
    vllm_qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)
    raw_gate = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=dtype)
    raw_beta = torch.randn(1, batch, heads, device="cuda", dtype=dtype)
    A_log = 0.1 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.1 * torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    state_indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    base_state = 0.01 * torch.randn(
        slots, heads, key_dim, value_dim, device="cuda", dtype=torch.float32
    )

    _, attention_gym_state = strided_state(base_state.shape, base_state)
    vllm_seed = base_state.transpose(-1, -2).contiguous()
    _, vllm_state = strided_state(vllm_seed.shape, vllm_seed)
    attention_gym_output = recurrent_kda_decode(
        attention_gym_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        attention_gym_state,
        state_indices,
        lower_bound=lower_bound,
    )
    vllm_output, _ = fused_recurrent_kda_packed_decode(
        vllm_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        vllm_state,
        state_indices,
    )
    torch.testing.assert_close(
        attention_gym_output.float(), vllm_output.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        attention_gym_state.transpose(-1, -2), vllm_state, atol=3e-2, rtol=3e-2
    )

    _, attention_gym_eager_state = strided_state(base_state.shape, base_state)
    _, vllm_eager_state = strided_state(vllm_seed.shape, vllm_seed)

    def attention_gym_eager():
        return recurrent_kda_decode(
            attention_gym_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            attention_gym_eager_state,
            state_indices,
            lower_bound=lower_bound,
        )

    def vllm_eager():
        return fused_recurrent_kda_packed_decode(
            vllm_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            lower_bound,
            vllm_eager_state,
            state_indices,
        )[0]

    attention_gym_eager_us = event_time_us(attention_gym_eager)
    vllm_eager_us = event_time_us(vllm_eager)

    _, attention_gym_graph_state = strided_state(base_state.shape, base_state)
    _, vllm_graph_state = strided_state(vllm_seed.shape, vllm_seed)

    def attention_gym_graph_call():
        return recurrent_kda_decode(
            attention_gym_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            attention_gym_graph_state,
            state_indices,
            lower_bound=lower_bound,
        )

    def vllm_graph_call():
        return fused_recurrent_kda_packed_decode(
            vllm_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            lower_bound,
            vllm_graph_state,
            state_indices,
        )[0]

    attention_gym_graph_us = graph_time_us(capture(attention_gym_graph_call))
    vllm_graph_us = graph_time_us(capture(vllm_graph_call))
    return (
        attention_gym_eager_us,
        vllm_eager_us,
        attention_gym_graph_us,
        vllm_graph_us,
    )


def main() -> None:
    print(
        "batch,heads,key_dim,value_dim,attention_gym_eager_us,vllm_eager_us,"
        "eager_speedup_pct,attention_gym_graph_us,vllm_graph_us,graph_speedup_pct"
    )
    for shape in SHAPES:
        repetitions = [benchmark_shape(*shape) for _ in range(REPEATS)]
        medians = tuple(statistics.median(values) for values in zip(*repetitions))
        attention_gym_eager_us, vllm_eager_us = medians[:2]
        attention_gym_graph_us, vllm_graph_us = medians[2:]
        results = (
            attention_gym_eager_us,
            vllm_eager_us,
            100 * (vllm_eager_us - attention_gym_eager_us) / vllm_eager_us,
            attention_gym_graph_us,
            vllm_graph_us,
            100 * (vllm_graph_us - attention_gym_graph_us) / vllm_graph_us,
        )
        print(
            ",".join(map(str, (*shape, *(round(value, 4) for value in results)))),
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
