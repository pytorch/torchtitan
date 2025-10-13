#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script to verify PyTorch compatibility shims are working correctly.
"""

import sys


def test_shims():
    """Test all compatibility shims."""
    print("Testing PyTorch compatibility shims...\n")

    # Test 1: Import torchtitan (which auto-installs shims)
    print("1. Testing torchtitan import...")
    try:
        import torchtitan  # noqa: F401

        print("   ✓ torchtitan imported successfully\n")
    except Exception as e:
        print(f"   ✗ Failed to import torchtitan: {e}\n")
        return False

    # Test 2: Consolidate HF safetensors module
    print("2. Testing torch.distributed.checkpoint._consolidate_hf_safetensors...")
    try:
        from torch.distributed.checkpoint._consolidate_hf_safetensors import (  # noqa: F401
            consolidate_safetensor_files,
            consolidate_safetensors_files_on_every_rank,
        )

        print("   ✓ Module and functions imported successfully\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False

    # Test 3: Checkpoint staging classes
    print("3. Testing torch.distributed.checkpoint.staging classes...")
    try:
        from torch.distributed.checkpoint.staging import (  # noqa: F401
            DefaultStager,
            StagingOptions,
        )

        print("   ✓ DefaultStager and StagingOptions imported successfully\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False

    # Test 4: Pipeline schedule class
    print("4. Testing torch.distributed.pipelining.schedules.ScheduleDualPipeV...")
    try:
        from torch.distributed.pipelining.schedules import (  # noqa: F401
            ScheduleDualPipeV,
        )

        print("   ✓ ScheduleDualPipeV imported successfully\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False

    # Test 5: Flex attention AuxOutput
    print("5. Testing torch.nn.attention.flex_attention.AuxOutput...")
    try:
        from torch.nn.attention.flex_attention import AuxOutput  # noqa: F401

        print("   ✓ AuxOutput imported successfully\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False

    # Test 6: Checkpoint wrapper with early_stop parameter
    print("6. Testing checkpoint_wrapper with early_stop parameter...")
    try:
        import torch.nn as nn
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        # Create a simple module and wrap it with early_stop parameter
        module = nn.Linear(10, 10)
        _ = checkpoint_wrapper(module, preserve_rng_state=False, early_stop=True)
        print("   ✓ checkpoint_wrapper accepts early_stop parameter\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False

    print("=" * 60)
    print("All compatibility shims are working correctly! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_shims()
    sys.exit(0 if success else 1)
