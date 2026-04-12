#!/bin/bash
# EP correctness check: same model, same input, same FSDP setup.
# Both runs use 4 GPUs with dp_shard=4.
# Only difference: ep=1 (experts replicated) vs ep=4 (experts sharded).
# With force_load_balance=True, losses should match within fp precision.
set -e

echo "=== EP=1 (4 GPUs, experts replicated) ==="
EP1_LOSS=$(PYTORCH_ALLOC_CONF="expandable_segments:True" \
  torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  scripts/ep_correctness.py --ep 1 2>&1 | grep "loss=" | grep -oP 'loss=\K[0-9.]+')
echo "EP=1 loss: ${EP1_LOSS}"

echo ""
echo "=== EP=4 (4 GPUs, experts sharded) ==="
EP4_LOSS=$(PYTORCH_ALLOC_CONF="expandable_segments:True" \
  torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  scripts/ep_correctness.py --ep 4 2>&1 | grep "loss=" | grep -oP 'loss=\K[0-9.]+')
echo "EP=4 loss: ${EP4_LOSS}"

echo ""
echo "=== Result ==="
echo "EP=1: ${EP1_LOSS}"
echo "EP=4: ${EP4_LOSS}"

python3 -c "
ep1, ep4 = float('${EP1_LOSS}'), float('${EP4_LOSS}')
diff = abs(ep1 - ep4)
print(f'Difference: {diff:.6f}')
if diff < 1e-2:
    print('PASS: EP correctness within tolerance')
else:
    print(f'FAIL: loss difference {diff} exceeds 1e-2')
    exit(1)
"
