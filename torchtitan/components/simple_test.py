# Import Dion optimizer components
import torch

try:
    from torchtitan.experiments.dion_optimizer.dion import (
        Dion,
        DionMixedPrecisionConfig,
    )
    from torchtitan.experiments.dion_optimizer.titan_dion import DionOptimizersContainer

    DION_AVAILABLE = True
    print("✓ Dion optimizer components imported")
except ImportError:
    DION_AVAILABLE = False
    print("✗ Dion optimizer components not available")
