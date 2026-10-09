#!/bin/bash
# Run the repros for ../README.md from this folder; each script's output goes to logs/<script>.txt.
# CPU repros run anywhere; the GPU repros need one CUDA GPU (the README's numbers are from one GB300).
# Usage: ./run_all.sh [cpu|gpu|all]   (default: all)
set -u
cd "$(dirname "$0")"
MODE=${1:-all}
mkdir -p logs

# run <log name> [ENV=VALUE ...] <script> [args...]: append the script's output to logs/<log name>.txt.
run() {
    local log="logs/$1.txt"
    shift
    echo "== $*" | tee -a "$log"
    env "$@" 2>&1 | grep -v -E "^[WI][0-9]{4} |UserWarning|warnings.warn|USDT" | tail -40 | tee -a "$log"
}

if [ "$MODE" = cpu ] || [ "$MODE" = all ]; then
    for f in ask01c_shared_norm_call_sites_cpu ask03_autograd_fn_param_base ask13_regional_partition_scaling \
             kb_make_fx_tensor_constant kb_hop_outer_attr_store kb_cache_under_fake_tensors kb_make_fx_int_specialization; do
        rm -f "logs/$f.txt"
        run "$f" CUDA_VISIBLE_DEVICES= python "$f.py"
    done
    rm -f logs/nla_sac_wrapped_region_recompile_cpu.txt
    run nla_sac_wrapped_region_recompile_cpu python nla_sac_wrapped_region_recompile.py cold --cpu
    run nla_sac_wrapped_region_recompile_cpu python nla_sac_wrapped_region_recompile.py warm --cpu
fi

if [ "$MODE" = gpu ] || [ "$MODE" = all ]; then
    for f in ask01_mix_order_guards ask01b_grad_mode_guard ask01c_shared_norm_call_sites ask01d_size1_specialization \
             ask02_cache_ignores_fake ask04_opaque_custom_op ask05_symbolic_hidden_dim ask06a_symbolic_small_dim \
             ask06a_small_k_in_tile ask06b_extern_gemv ask06c_int64_indexing_symbolic ask07_cat_lowering \
             ask09_sinkhorn_transposed_reads ask11_inline_recompute ask12_masked_loads_symbolic_T \
             ask12_standalone_masked_kernels kb_maybe_mark_dynamic_traced nla_remat_checkpoint_wrapped_regions \
             obs_whole_graph_vs_region; do
        rm -f "logs/$f.txt"
        run "$f" python "$f.py"
    done
    rm -f logs/nla_sac_wrapped_region_recompile.txt logs/ask08_complex_ops.txt logs/kb_fma_bitwise.txt
    run nla_sac_wrapped_region_recompile python nla_sac_wrapped_region_recompile.py cold
    run nla_sac_wrapped_region_recompile python nla_sac_wrapped_region_recompile.py warm
    # Ask 8: Inductor prints the complex-codegen warning only on a cold compile.
    run ask08_complex_ops TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python ask08_complex_ops.py
    # Known behavior (FP contraction): bitwise once FP contraction is off; TRITON_DEFAULT_FP_FUSION has no effect.
    for e in 0 1; do
        run kb_fma_bitwise EMULATE_PRECISION_CASTS=$e TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python kb_fma_bitwise.py
    done
    run kb_fma_bitwise TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TRITON_DEFAULT_FP_FUSION=0 python kb_fma_bitwise.py
fi
