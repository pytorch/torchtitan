// Copyright (c) 2026, trainstation team
// The following code is copied from https://github.com/open-lm-engine/lm-engine

// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <metal_stdlib>

using namespace metal;

static inline float _sigmoid(float g) {
    return 1.0f / (1.0f + exp(-g));
}

template <typename T>
static inline void _swiglu_forward(device const T *g, device const T *u, device T *y, uint id) {
    float _g = float(g[id]);
    y[id] = T(float(u[id]) * _g * _sigmoid(_g));
}

kernel void swiglu_forward_fp32(device const float *g [[buffer(0)]],
                                device const float *u [[buffer(1)]],
                                device float *y [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    _swiglu_forward(g, u, y, id);
}

kernel void swiglu_forward_fp16(device const half *g [[buffer(0)]],
                                device const half *u [[buffer(1)]],
                                device half *y [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    _swiglu_forward(g, u, y, id);
}

kernel void swiglu_forward_bf16(device const bfloat *g [[buffer(0)]],
                                device const bfloat *u [[buffer(1)]],
                                device bfloat *y [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    _swiglu_forward(g, u, y, id);
}
