#!/bin/bash
# Run DeepEP tests

set -e

NGPU=${NGPU:-2}
SKIP_UNIT=${SKIP_UNIT:-0}
SKIP_DISTRIBUTED=${SKIP_DISTRIBUTED:-0}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ngpu) NGPU="$2"; shift 2 ;;
        --skip-unit) SKIP_UNIT=1; shift ;;
        --skip-distributed) SKIP_DISTRIBUTED=1; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ngpu N              GPUs for distributed tests (default: 2)"
            echo "  --skip-unit           Skip unit tests"
            echo "  --skip-distributed    Skip distributed tests"
            echo "  -h, --help            Show help"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "DeepEP Tests"
echo "========================================"
echo ""

# Unit tests
if [ "$SKIP_UNIT" -eq 0 ]; then
    echo "Running helper function tests..."
    python -m unittest tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher.TestHelperFunctions -v
    [ $? -eq 0 ] && echo "✓ Helper function tests passed" || exit $?
    echo ""

    echo "Running unit tests..."
    python -m unittest tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher.TestUnit -v
    [ $? -eq 0 ] && echo "✓ Unit tests passed" || exit $?
    echo ""
else
    echo "Skipping unit tests"
fi

# Distributed tests
if [ "$SKIP_DISTRIBUTED" -eq 0 ]; then
    echo "Running distributed tests..."

    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "⚠ Skipping: CUDA not available"
        exit 0
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -lt "$NGPU" ]; then
        echo "⚠ Skipping: Need $NGPU GPUs, found $GPU_COUNT"
        exit 0
    fi

    torchrun --nproc_per_node=$NGPU -m tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher
    [ $? -eq 0 ] && echo "✓ Distributed tests passed" || exit $?
else
    echo "Skipping distributed tests"
fi

echo ""
echo "========================================"
echo "All tests passed!"
echo "========================================"
