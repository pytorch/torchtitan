// Copyright (c) 2026, trainstation team
// The following code is copied from https://github.com/open-lm-engine/lm-engine

// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

// PyTorch C++ extension header — gives us torch::Tensor, TORCH_CHECK, pybind11 bindings, etc.
#include <torch/extension.h>

// PyTorch's Metal shader utilities — provides MetalShaderLibrary (compiles Metal source,
// caches pipelines) and MetalKernelFunction (encodes and dispatches compute kernels).
// This handles all the low-level Metal boilerplate: device selection, library compilation,
// pipeline creation, command buffer/encoder management, and stream synchronization.
#include <ATen/native/mps/MetalShaderLibrary.h>

#include <fstream>
#include <sstream>

// read_file: reads a file from disk into a string.
static std::string read_file(const std::string &path) {
    std::ifstream f(path);
    TORCH_CHECK(f.is_open(), "failed to open ", path);
    std::stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

// read_shader_sources: reads and concatenates all .metal shader files.
// We keep shaders in separate .metal files (rather than embedding as string
// literals) so they can be edited and syntax-highlighted independently.
static std::string read_shader_sources() {
    // __FILE__ expands to the absolute path of this .mm file at compile time.
    // We strip the filename to get the directory, then read forward.metal and backward.metal.
    std::string dir(__FILE__);
    dir = dir.substr(0, dir.rfind('/'));

    return read_file(dir + "/forward.metal") + "\n" + read_file(dir + "/backward.metal");
}

// get_library: returns a singleton MetalShaderLibrary that lazily compiles the shader.
// MetalShaderLibrary is PyTorch's built-in abstraction over Metal's compilation pipeline:
//   - reads Metal source code (as a std::string)
//   - compiles it into a MTLLibrary (collection of GPU functions)
//   - caches MTLComputePipelineState objects per kernel function name
// DynamicMetalShaderLibrary compiles immediately on construction (vs. lazily).
static at::native::mps::MetalShaderLibrary &get_library() {
    // Using a function-local static ensures thread-safe one-time initialization (C++11 guarantee).
    static at::native::mps::DynamicMetalShaderLibrary lib(read_shader_sources());
    return lib;
}

// swiglu_forward_mps: the main entry point called from Python.
// Dispatches the Metal kernel to compute y = u * g * sigmoid(g) element-wise.
void _swiglu_forward_mps(const torch::Tensor &g, const torch::Tensor &u, torch::Tensor &y) {
    // Validate that all tensors live on the MPS device (Apple GPU)
    TORCH_CHECK(g.is_mps() && u.is_mps() && y.is_mps(), "all tensors must be on MPS device");
    // Metal kernels index by thread ID into flat memory, so tensors must be contiguous
    TORCH_CHECK(g.is_contiguous() && u.is_contiguous() && y.is_contiguous(), "all tensors must be contiguous");

    // Pick the kernel function name based on dtype.
    // These names match the `kernel void swiglu_forward_float(...)` and
    // `kernel void swiglu_forward_half(...)` functions in forward.metal.
    std::string kernel_name;
    if (g.scalar_type() == torch::kFloat) {
        kernel_name = "swiglu_forward_fp32";
    } else if (g.scalar_type() == torch::kHalf) {
        kernel_name = "swiglu_forward_fp16";
    } else if (g.scalar_type() == torch::kBFloat16) {
        kernel_name = "swiglu_forward_bf16";
    } else {
        TORCH_CHECK(false, "unsupported dtype for MPS swiglu");
    }

    // getKernelFunction returns a MetalKernelFunction — a wrapper around a compiled
    // MTLComputePipelineState with helper methods for argument binding and dispatch.
    // The pipeline is created and cached on first call for each kernel_name.
    auto kernel = get_library().getKernelFunction(kernel_name);

    // runCommandBlock obtains a command buffer from PyTorch's MPS stream (ensuring
    // correct ordering with other PyTorch MPS ops) and commits it when the block exits.
    // This replaces the manual stream->commandBuffer() + stream->commit() pattern.
    kernel->runCommandBlock([&]() {
        // startEncoding creates a MTLComputeCommandEncoder on the command buffer
        // and sets the compute pipeline state (which kernel to run).
        kernel->startEncoding();
        // setArg binds each tensor's underlying Metal buffer to the corresponding
        // buffer index (matches [[buffer(0)]], [[buffer(1)]], [[buffer(2)]] in .metal).
        // Internally calls [encoder setBuffer:getMTLBufferStorage(t) offset:... atIndex:idx].
        kernel->setArg(0, g);
        kernel->setArg(1, u);
        kernel->setArg(2, y);
        // dispatch launches one GPU thread per element. Metal automatically divides
        // the 1D grid into threadgroups sized to the hardware limit (typically 1024).
        // This is analogous to cudaLaunchKernel(grid=numel, block=maxThreadsPerBlock).
        kernel->dispatch(static_cast<uint64_t>(g.numel()));
    });
}

// swiglu_backward_mps: backward pass entry point called from Python.
// Computes gradients: dg = dy * u * (sigmoid(g) + silu(g) * (1 - sigmoid(g)))
//                     du = dy * silu(g)
void _swiglu_backward_mps(
    const torch::Tensor &g, const torch::Tensor &u, const torch::Tensor &dy, torch::Tensor &dg, torch::Tensor &du) {
    TORCH_CHECK(g.is_mps() && u.is_mps() && dy.is_mps() && dg.is_mps() && du.is_mps(),
                "all tensors must be on MPS device");
    TORCH_CHECK(
        g.is_contiguous() && u.is_contiguous() && dy.is_contiguous() && dg.is_contiguous() && du.is_contiguous(),
        "all tensors must be contiguous");

    std::string kernel_name;
    if (g.scalar_type() == torch::kFloat) {
        kernel_name = "swiglu_backward_fp32";
    } else if (g.scalar_type() == torch::kHalf) {
        kernel_name = "swiglu_backward_fp16";
    } else if (g.scalar_type() == torch::kBFloat16) {
        kernel_name = "swiglu_backward_bf16";
    } else {
        TORCH_CHECK(false, "unsupported dtype for MPS swiglu backward");
    }

    auto kernel = get_library().getKernelFunction(kernel_name);

    kernel->runCommandBlock([&]() {
        kernel->startEncoding();
        kernel->setArg(0, g);
        kernel->setArg(1, u);
        kernel->setArg(2, dy);
        kernel->setArg(3, dg);
        kernel->setArg(4, du);
        kernel->dispatch(static_cast<uint64_t>(g.numel()));
    });
}

// pybind11 module definition — this is what torch.utils.cpp_extension.load() looks for.
// It creates a Python module and exposes our C++ functions as callable Python functions.
// TORCH_EXTENSION_NAME is a macro set by the build system to the module name we specified.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_swiglu_forward_mps", &_swiglu_forward_mps, "SwiGLU forward (MPS)");
    m.def("_swiglu_backward_mps", &_swiglu_backward_mps, "SwiGLU backward (MPS)");
}
