#!/bin/bash
# Run the repros for README.md; each script's output goes to logs/<script>.txt. All need one CUDA GPU (Blackwell for
# the GEMM tiles; the README's numbers are from one GB300).
#   gpu: pure-torch repros.
#   ext: repros that need out-of-tree builds: GMM_EXT_DIR (CUTLASS extensions, ask 2) and
#        CUTLASS_DSL_SITE (a site-packages dir with cutlass-dsl 4.8, ask 7).
# Usage: ./run_all.sh [gpu|ext|all]   (default: gpu)
set -u
cd "$(dirname "$0")/repros"
MODE=${1:-gpu}
mkdir -p ../logs

# run <log name> [ENV=VALUE ...] <script> [args...]: append the script's output to ../logs/<log name>.txt.
run() {
    local log="../logs/$1.txt"
    shift
    echo "== $*" | tee -a "$log"
    env "$@" 2>&1 | grep -v -E "^[WIV][0-9]{4} |UserWarning|warnings.warn|warn_once|USDT|Warning:|AUTOTUNE|^ +triton_grouped_mm_|^Autotune Choices Stats|^\{.num_choices|^strides: |^dtypes: " | tail -60 | tee -a "$log"
}

if [ "$MODE" = gpu ] || [ "$MODE" = all ]; then
    for f in ask01_grouped_mm_small_groups ask03_gather_and_swiglu ask06_triton_template_small_m \
             ask08_dynamic_autotune_first_size ask09_dynamic_bounds_mask ask10_dispatch_codegen ask11_router_row_gather ask12_split_reduction_size_hint \
             kb_profiler_slows_graph_replay; do
        rm -f "../logs/$f.txt"
        run "$f" python "$f.py"
    done
    # Ask 4: same script without and with PDL; cold caches so each run compiles its own kernels.
    rm -f ../logs/ask04_pdl.txt
    for pdl in 0 1; do
        run ask04_pdl TORCHINDUCTOR_ENABLE_PDL=$pdl TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python ask04_pdl.py
    done
    # Ask 5: extern ATen GEMM vs Inductor's Triton template (the triton run also prints the fusion-log reasons).
    rm -f ../logs/ask05_grouped_mm_template_fusion.txt
    run ask05_grouped_mm_template_fusion python ask05_grouped_mm_template_fusion.py default
    run ask05_grouped_mm_template_fusion python ask05_grouped_mm_template_fusion.py triton


fi

if [ "$MODE" = ext ] || [ "$MODE" = all ]; then
    rm -f ../logs/ask02_ptr_array_vs_moe_kernel.txt ../logs/ask07_cutedsl_template.txt
    # Ask 2: two runs on a shared GPU; the README quotes the min.
    for _ in 1 2; do
        run ask02_ptr_array_vs_moe_kernel GMM_EXT_DIR="$GMM_EXT_DIR" python ask02_ptr_array_vs_moe_kernel.py
    done
    run ask07_cutedsl_template PYTHONPATH="$CUTLASS_DSL_SITE:${PYTHONPATH:-}" python ask07_cutedsl_template.py
fi
