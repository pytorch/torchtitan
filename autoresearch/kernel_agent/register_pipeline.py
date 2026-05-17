"""Import hook: run this before torchtitan.train to register the fused_kernels pipeline.

Usage with torchrun:
  torchrun ... -m autoresearch.kernel_agent.register_pipeline torchtitan.train --module ... --config ...
"""

import sys
import runpy

if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
    _target_module = sys.argv.pop(1)
else:
    print("Usage: python -m autoresearch.kernel_agent.register_pipeline <module> [args...]")
    sys.exit(1)

# Import AFTER argv cleanup so that any transitive imports
# that touch argv see the cleaned-up version.
import autoresearch.kernel_agent.graph_pass  # noqa: F401, E402

if __name__ == "__main__":
    runpy.run_module(_target_module, run_name="__main__", alter_sys=True)
