# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys

import pytest


def test_common_attention_imports_without_attention_gym() -> None:
    """Shared metadata and linear backends can import without Attention Gym."""
    script = r"""
import importlib
import importlib.abc
import sys

class BlockAttentionGym(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "attn_gym" or fullname.startswith("attn_gym."):
            raise ModuleNotFoundError("blocked dependency", name="attn_gym")
        return None

sys.meta_path.insert(0, BlockAttentionGym())
import torchtitan.config.transform
import torchtitan.models.common
from torchtitan.models.common.attention import LinearAttentionMetadata
from torchtitan.models.common.attention import cp_gdn, cp_kda, gdn, kda
from torchtitan.models.common.attn_gym import _MissingKernel
from spmd_types._local_registration import _LOCAL_AUTOGRAD_FUNCTIONS

assert LinearAttentionMetadata is kda.LinearAttentionMetadata
assert LinearAttentionMetadata.__module__ == kda.__name__
assert _MissingKernel not in _LOCAL_AUTOGRAD_FUNCTIONS
assert "attn_gym" not in sys.modules

for kernel in (
    kda.chunk_kda,
    gdn.chunk_gdn,
    cp_kda.context_parallel_kda,
    cp_gdn.context_parallel_gdn,
    cp_kda.ContextParallelRouting.from_fragments,
):
    try:
        kernel()
    except ModuleNotFoundError as error:
        assert error.name == "attn_gym"
        assert "attn-gym[linear]==0.0.16" in str(error)
    else:
        raise AssertionError("A missing Attention Gym kernel ran")

for model_name in (
    "llama3", "qwen3", "deepseek_v3", "deepseek_v4", "gpt_oss",
    "muse_glimmer", "qwen3_5", "kimi_k3",
):
    model = importlib.import_module(f"torchtitan.models.{model_name}")
    model.build_model_config("debugmodel", seq_len=128)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_attention_gym_facade_loads_only_requested_exports() -> None:
    """Backend exports keep their identity and load on demand."""
    script = r"""
from types import ModuleType
import torchtitan.models.common.attn_gym as facade

# The common package can cache these exports while importing KDA.
facade.__dict__.pop("l2norm", None)
facade.__dict__.pop("gather_attn", None)

linear = ModuleType("linear_backend")
sparse = ModuleType("sparse_backend")
linear.l2norm = object()
sparse.gather_attn = object()
modules = {
    "attn_gym.linear.kda.fwd.triton.l2norm_fwd": linear,
    "attn_gym.sparse.gather_attn": sparse,
}
requests = []

def load_module(name):
    requests.append(name)
    return modules[name]

facade.import_module = load_module
assert requests == []
from torchtitan.models.common.attn_gym import l2norm
assert l2norm is linear.l2norm
from torchtitan.models.common.attn_gym import l2norm
assert len(requests) == 1
from torchtitan.models.common.attn_gym import gather_attn
assert gather_attn is sparse.gather_attn
assert requests == list(modules)
try:
    facade.unknown_kernel
except AttributeError:
    pass
else:
    raise AssertionError("An unknown kernel got a placeholder")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["gdn", "kda"])
@pytest.mark.parametrize("missing_package", ["attn_gym", "cutlass"])
def test_linear_attention_reports_missing_dependency(
    backend: str, missing_package: str
) -> None:
    """Building a kernel fails when its package is missing or broken."""
    script = r"""
import importlib
import importlib.abc
import sys
from types import ModuleType

backend, missing_package = sys.argv[1:]
if missing_package != "attn_gym":
    package = ModuleType("attn_gym")
    package.__path__ = []
    sys.modules["attn_gym"] = package

class BlockAttentionGym(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "attn_gym" or fullname.startswith("attn_gym."):
            raise ModuleNotFoundError("blocked dependency", name=missing_package)
        return None

sys.meta_path.insert(0, BlockAttentionGym())
try:
    module = importlib.import_module(f"torchtitan.models.common.attention.{backend}")
    kernel = getattr(module, f"{backend.upper()}Kernel")
    kernel.Config().build()
except ModuleNotFoundError as error:
    assert error.name == missing_package
    if missing_package == "attn_gym":
        assert "attn-gym[linear]==0.0.16" in str(error)
    else:
        assert str(error) == "blocked dependency"
else:
    raise AssertionError("Linear attention built without its dependency")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, backend, missing_package],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("other_kernel_available", [False, True])
def test_attention_gym_registration_keeps_real_functions(
    other_kernel_available: bool,
) -> None:
    """Only real kernel classes reach the SPMD autograd registry."""
    script = r"""
import importlib
import sys
from unittest.mock import call, Mock

import spmd_types as spmd
import torch
from torchtitan.models.common import attn_gym
from torchtitan.models.common.attention import kda

class Kernel(torch.autograd.Function):
    pass

class OtherKernel(torch.autograd.Function):
    pass

attn_gym._ShortConv = Kernel
attn_gym._ConfiguredShortConv = (
    OtherKernel if sys.argv[1] == "True" else attn_gym._MissingKernel
)
register = Mock()
spmd.register_local_autograd_function = register
importlib.reload(kda)
assert register.call_args_list.count(call(Kernel)) == 1
assert call(attn_gym._MissingKernel) not in register.call_args_list
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(other_kernel_available)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dist_moe_modules_defer_optional_dependency_import() -> None:
    """Importing Dist-MoE integration modules does not load the backend."""
    script = r"""
import importlib.abc
import sys

class BlockDistMoe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "dist_moe" or fullname.startswith("dist_moe."):
            raise ModuleNotFoundError("blocked optional import", name=fullname)
        return None

sys.meta_path.insert(0, BlockDistMoe())
import torchtitan.config.transform
import torchtitan.config.transform.dist_moe
import torchtitan.models.common.dist_moe
import torchtitan.models.common.lora
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
