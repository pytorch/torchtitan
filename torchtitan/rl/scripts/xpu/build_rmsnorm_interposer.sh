#!/bin/bash
# Build ONLY the patched LayerNormKernels.cpp as an LD_PRELOAD interposer.
#
# Why not rebuild PyTorch: the installed torch in the `monarch` env is a wheel
# (2.12.0+xpu, Jul 27) and every RL measurement to date was taken on it. A source
# rebuild is multi-hour AND would produce a different torch, invalidating the
# comparison baseline. So instead we exploit two facts about the wheel:
#
#   1. `at::native::xpu::rms_norm_backward_kernel` is an EXPORTED dynamic symbol
#      in libtorch_xpu.so, and its call site goes through the PLT
#      (R_X86_64_JUMP_SLOT), so it is interposable by LD_PRELOAD.
#   2. The installed symbol's mangled signature matches today's source exactly:
#      (Tensor const&, Tensor const&, Tensor const&, Tensor const&, long, long,
#       Tensor*, Tensor*)
#      so a freshly compiled definition is ABI-compatible with the existing call.
#
# Preloading a .so that defines the same symbol makes the dynamic linker resolve
# the PLT slot to OUR copy. The rest of libtorch_xpu.so is untouched, and nothing
# in the conda env is modified -- remove the LD_PRELOAD and you are back to the
# stock wheel bit-for-bit.
#
# Caveat this build must respect: LayerNormKernels.cpp also defines
# `layer_norm_backward_kernel` and the forward kernels. Our .so will export those
# too and will therefore interpose them as well. That is acceptable here (same
# source, same flags) but it does mean this is a validation vehicle, not a
# shipping mechanism -- the actual fix is the torch-xpu-ops commit.
#
# Usage: bash build_rmsnorm_interposer.sh [output_dir]
# Override the torch-xpu-ops checkout with XPU_OPS=/path/to/torch-xpu-ops.

XPU_OPS=${XPU_OPS:-$HOME/git/torch-xpu-ops}
OUT_DIR="${1:-$HOME/git/torchtitan/torchtitan/rl/rmsnorm_interposer}"
SRC="$XPU_OPS/src/ATen/native/xpu/sycl/LayerNormKernels.cpp"

# setvars.sh and conda's init are not written to survive `set -euo pipefail`
# (unbound vars, non-zero returns), so source them first and enable strict mode
# only afterwards.
source /opt/aurora/26.26.0/oneapi/setvars.sh >/dev/null 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate monarch

set -euo pipefail

TORCH_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_INC="$TORCH_DIR/include"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "=== build config ==="
echo "torch      : $(python -c 'import torch; print(torch.__version__)')"
echo "torch dir  : $TORCH_DIR"
echo "source     : $SRC"
echo "xpu-ops    : $(cd $XPU_OPS && git log --oneline -1)"
echo "compiler   : $(icpx --version | head -1)"
echo "output     : $OUT_DIR/libinterpose_layernorm.so"
echo

# Include order matters: torch-xpu-ops' own src/ must come first so that
# `comm/SYCLContext.h` and `ATen/native/xpu/sycl/Norm.h` resolve to the patched
# tree rather than anything in the wheel.
INCLUDES=(
  -I"$XPU_OPS/src"
  -I"$TORCH_INC"
  -I"$TORCH_INC/torch/csrc/api/include"
  -I"$TORCH_INC/TH"
  -I"$TORCH_INC/THC"
)

# Mirror cmake/BuildFlags.cmake set_build_flags() for the GNU host-compiler path.
# AOT target is pvc only (Aurora Max 1550); the stock wheel targets a wider list,
# but device code for other parts is unaffected since we only replace this TU.
SYCL_FLAGS=(
  -fsycl
  -fsycl-targets=spir64_gen,spir64
  -fno-sycl-unnamed-lambda
  -sycl-std=2020
  -foffload-fp32-prec-div
  -foffload-fp32-prec-sqrt
  -fno-fast-math
  -ffp-contract=fast
  -std=c++20
  -Wno-absolute-value
  -Wno-sign-compare
  -Wno-interference-size
)

DEFINES=(
  -DSYCL_COMPILER_VERSION=20250302
  -DUSE_XPU=1
  -DAT_PER_OPERATOR_HEADERS
  -DC10_USING_CUSTOM_GENERATED_MACROS
  -DTORCH_XPU_BUILD_MAIN_LIB
  -DUSE_C10D_XCCL
)

# _GLIBCXX_USE_CXX11_ABI must match the wheel or every std::string crossing the
# boundary corrupts. Read it off the installed torch rather than guessing.
CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
echo "=== _GLIBCXX_USE_CXX11_ABI = $CXX11_ABI (read from installed torch) ==="
DEFINES+=(-D_GLIBCXX_USE_CXX11_ABI=$CXX11_ABI)

echo "=== compiling (this takes several minutes: AOT device codegen for pvc) ==="
time icpx \
  "${SYCL_FLAGS[@]}" \
  "${DEFINES[@]}" \
  "${INCLUDES[@]}" \
  -fPIC -O2 -shared \
  -Xs "-device pvc -options -cl-poison-unsupported-fp64-kernels" \
  "$SRC" \
  -L"$TORCH_DIR/lib" \
  -ltorch_xpu -lc10_xpu -ltorch_cpu -lc10 -ltorch \
  -Wl,-rpath,"$TORCH_DIR/lib" \
  -o libinterpose_layernorm.so \
  2>&1 | tail -40

echo
echo "=== verify the interposing symbol is exported ==="
nm -D --defined-only libinterpose_layernorm.so | grep rms_norm_backward_kernel | c++filt

echo
echo "=== compare against the installed wheel's mangled name ==="
MINE=$(nm -D --defined-only libinterpose_layernorm.so | grep -o "_ZN2at6native3xpu24rms_norm_backward_kernel[A-Za-z0-9_]*" | head -1)
THEIRS=$(nm -D --defined-only "$TORCH_DIR/lib/libtorch_xpu.so" | grep -o "_ZN2at6native3xpu24rms_norm_backward_kernel[A-Za-z0-9_]*" | head -1)
echo "ours   : $MINE"
echo "wheel  : $THEIRS"
if [ "$MINE" = "$THEIRS" ] && [ -n "$MINE" ]; then
  echo "MATCH -- LD_PRELOAD will interpose correctly"
else
  echo "MISMATCH -- LD_PRELOAD will NOT take effect; do not trust any test result"
  exit 1
fi
