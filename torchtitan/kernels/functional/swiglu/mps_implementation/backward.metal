// Copyright (c) 2026, trainstation team
// The following code is copied from https://github.com/open-lm-engine/lm-engine

// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

template <typename T>
static inline void _swiglu_backward(device const T *g, device const T *u, device const T *dy,
                                   device T *dg, device T *du, uint id) {
    float _g = float(g[id]);
    float _u = float(g[id]);
    float _dy = float(dy[id]);

    float g_sigmoid = _sigmoid(_g);
    float g_silu = _g * g_sigmoid;
    float _dg = _dy * _u * (g_sigmoid + g_silu * (1.0f - g_sigmoid));
    float _du = _dy * g_silu;

    dg[id] = T(_dg);
    du[id] = T(_du);
}

kernel void swiglu_backward_fp32(device const float *g [[buffer(0)]],
                                 device const float *u [[buffer(1)]],
                                 device const float *dy [[buffer(2)]],
                                 device float *dg [[buffer(3)]],
                                 device float *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    _swiglu_backward(g, u, dy, dg, du, id);
}

kernel void swiglu_backward_fp16(device const half *g [[buffer(0)]],
                                 device const half *u [[buffer(1)]],
                                 device const half *dy [[buffer(2)]],
                                 device half *dg [[buffer(3)]],
                                 device half *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    _swiglu_backward(g, u, dy, dg, du, id);
}

kernel void swiglu_backward_bf16(device const bfloat *g [[buffer(0)]],
                                 device const bfloat *u [[buffer(1)]],
                                 device const bfloat *dy [[buffer(2)]],
                                 device bfloat *dg [[buffer(3)]],
                                 device bfloat *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    _swiglu_backward(g, u, dy, dg, du, id);
}
