#!/bin/bash
# oneAPI 2026.1 toolchain for the torch 2.14.0+xpu RL stack on Aurora; sourced by
# every launcher in this directory (override with ONEAPI_ENV_SCRIPT).
#
# torch 2.14.0+xpu ships the oneAPI 2026.1 SYCL runtime (libsycl.so.9). Never mix
# it with a oneAPI 2025.x environment (libsycl.so.8): Triton builds its launcher
# with the icpx on PATH, and two SYCL runtimes in one process segfault in
# sycl::device::get_backend().
module reset 2>/dev/null || true
module load oneapi/release/2026.1.0 gcc/14.3.0 2>/dev/null || true
if ! icpx --version 2>/dev/null | head -1 | grep -q "2026\.1"; then
    echo "ERROR: env_torch214.sh expected icpx 2026.1, got: $(icpx --version 2>&1 | head -1)"
fi
# The 2026.1 module sets ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE (one 128 GB device per
# card, so ZE_AFFINITY_MASK=N picks card N and masks >= 6 see no device), plus
# ZE_ENABLE_API_TRACING=1 and ZE_ENABLE_PCI_ID_DEVICE_ORDER=1. Every launcher's tile
# masks assume FLAT (12 x 64 GB tiles per node). Under COMPOSITE, co-located
# processes OOM each other and sampled outputs are not run-to-run reproducible.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_ENABLE_API_TRACING ZE_ENABLE_PCI_ID_DEVICE_ORDER
