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
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import LinearAttentionMetadata
from torchtitan.models.common.attention import (
    cp_gdn,
    cp_kda,
    cp_linear_attention,
    gdn,
    kda,
    linear_attention,
)
from torchtitan.models.common.attn_gym import _MissingKernel
from spmd_types._local_registration import _LOCAL_AUTOGRAD_FUNCTIONS

assert LinearAttentionMetadata is linear_attention.LinearAttentionMetadata
assert LinearAttentionMetadata.__module__ == linear_attention.__name__
assert _MissingKernel not in _LOCAL_AUTOGRAD_FUNCTIONS
assert "attn_gym" not in sys.modules

for kernel in (
    kda.chunk_kda,
    gdn.chunk_gdn,
    cp_kda.context_parallel_kda,
    cp_gdn.context_parallel_gdn,
    cp_linear_attention.ContextParallelRouting.from_fragments,
):
    try:
        kernel()
    except ModuleNotFoundError as error:
        assert error.name == "attn_gym"
        assert "attn-gym[linear,cudnn]==0.0.16" in str(error)
    else:
        raise AssertionError("A missing Attention Gym kernel ran")

for model_name in (
    "llama3", "qwen3", "deepseek_v3", "deepseek_v4", "gpt_oss",
    "muse_glimmer", "qwen3_5", "kimi_k2_7", "kimi_k3",
):
    model = importlib.import_module(f"torchtitan.models.{model_name}")
    model.build_model_config(
        "debugmodel", seq_len=128, parallelism=ParallelismConfig()
    )
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
        assert "attn-gym[linear,cudnn]==0.0.16" in str(error)
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
from torchtitan.models.common.attention import linear_attention

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
importlib.reload(linear_attention)
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


@pytest.mark.parametrize("missing_backend", ["cutlass", "triton"])
def test_common_attention_imports_with_broken_attention_gym_backend(
    missing_backend: str,
) -> None:
    """An installed attn_gym whose kernel backend is missing stays importable."""
    script = r"""
import importlib
import importlib.abc
import sys
from types import ModuleType

missing_backend = sys.argv[1]

# attn_gym itself imports, as it does on a platform without wheels for the
# backend, but every attn_gym module fails on the backend import.
package = ModuleType("attn_gym")
package.__path__ = []
sys.modules["attn_gym"] = package


class BreakAttentionGymBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("attn_gym."):
            raise ModuleNotFoundError("blocked dependency", name=missing_backend)
        return None


sys.meta_path.insert(0, BreakAttentionGymBackend())

import torchtitan.config.transform
import torchtitan.models.common
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    cp_gdn,
    cp_kda,
    cp_linear_attention,
    gdn,
    kda,
)
from torchtitan.models.common.attn_gym import _MissingKernel, is_missing_kernel
from spmd_types._local_registration import _LOCAL_AUTOGRAD_FUNCTIONS

assert _MissingKernel not in _LOCAL_AUTOGRAD_FUNCTIONS
assert not any(is_missing_kernel(f) for f in _LOCAL_AUTOGRAD_FUNCTIONS)
assert is_missing_kernel(kda.chunk_kda) and kda.chunk_kda is not _MissingKernel

for kernel in (
    kda.chunk_kda,
    gdn.chunk_gdn,
    cp_kda.context_parallel_kda,
    cp_gdn.context_parallel_gdn,
    cp_linear_attention.ContextParallelRouting.from_fragments,
    kda.KDAKernel.Config().build,
    gdn.GDNKernel.Config().build,
):
    try:
        kernel()
    except ModuleNotFoundError as error:
        assert error.name == missing_backend, error
        assert str(error) == "blocked dependency", error
    else:
        raise AssertionError("A kernel with a missing backend ran")

for model_name in (
    "llama3", "qwen3", "deepseek_v3", "deepseek_v4", "gpt_oss",
    "muse_glimmer", "qwen3_5", "kimi_k2_7", "kimi_k3",
):
    model = importlib.import_module(f"torchtitan.models.{model_name}")
    model.build_model_config(
        "debugmodel", seq_len=128, parallelism=ParallelismConfig()
    )
"""
    result = subprocess.run(
        [sys.executable, "-c", script, missing_backend],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_attention_gym_internal_module_errors_still_raise() -> None:
    """A missing attn_gym module is breakage, not a missing backend."""
    script = r"""
import importlib
import importlib.abc
import sys
from types import ModuleType

package = ModuleType("attn_gym")
package.__path__ = []
sys.modules["attn_gym"] = package


class MoveAttentionGymModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("attn_gym."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, MoveAttentionGymModules())

try:
    import torchtitan.models.common.attention.linear_attention
except ModuleNotFoundError as error:
    assert error.name is not None and error.name.startswith("attn_gym."), error
else:
    raise AssertionError("A missing attn_gym module was treated as a missing backend")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_attention_gym_mixed_backend_availability() -> None:
    """Exports whose backend imports stay real; only the failing ones defer."""
    script = r"""
import importlib
from types import ModuleType
from unittest.mock import call, Mock

import spmd_types as spmd
import torch
import torchtitan.models.common.attn_gym as facade
from torchtitan.models.common.attention import linear_attention


class Kernel(torch.autograd.Function):
    pass


class OtherKernel(torch.autograd.Function):
    pass


short_conv = ModuleType("attn_gym.linear.short_conv.cute")
short_conv._ShortConv = Kernel
short_conv._ConfiguredShortConv = OtherKernel
linear = ModuleType("attn_gym.linear")
linear.KernelOptions = object()
available = {
    "attn_gym.linear.short_conv.cute": short_conv,
    "attn_gym.linear": linear,
}


def load_module(name):
    if name in available:
        return available[name]
    backend = "cutlass" if name == "attn_gym.linear.kda.impl.fused" else "triton"
    raise ModuleNotFoundError("blocked dependency", name=backend)


facade.import_module = load_module
facade._BACKEND_ERRORS.clear()
facade._BACKEND_KERNELS.clear()
for name in ("_ShortConv", "_ConfiguredShortConv", "_L2Norm", "KernelOptions", "_ChunkKDA"):
    facade.__dict__.pop(name, None)
register = Mock()
spmd.register_local_autograd_function = register

importlib.reload(linear_attention)

assert register.call_args_list.count(call(Kernel)) == 1
assert register.call_args_list.count(call(OtherKernel)) == 1
assert call(facade._MissingKernel) not in register.call_args_list
assert linear_attention._ShortConv is Kernel
assert facade.is_missing_kernel(linear_attention._L2Norm)
assert linear_attention._L2Norm is not facade._MissingKernel
assert linear_attention._L2Norm.__name__ == "_MissingKernel[attn_gym.linear.kda.fwd.triton.l2norm_fwd]"
assert set(facade._BACKEND_ERRORS) == {"attn_gym.linear.kda.fwd.triton.l2norm_fwd"}


def backend_of(kernel):
    try:
        kernel()
    except ModuleNotFoundError as error:
        assert str(error) == "blocked dependency", error
        return error.name
    raise AssertionError("A kernel with a missing backend ran")


# Each placeholder reports its own module's backend, whichever failed first.
assert backend_of(linear_attention._L2Norm) == "triton"
assert backend_of(facade._ChunkKDA) == "cutlass"
assert backend_of(facade._ChunkKDA.apply) == "cutlass"
assert backend_of(lambda: facade.require_attn_gym(facade._ChunkKDA)) == "cutlass"
assert backend_of(lambda: facade.require_attn_gym(Kernel, linear_attention._L2Norm)) == "triton"
facade.require_attn_gym(Kernel, OtherKernel)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "broken_modules, missing_backend",
    [
        (("attn_gym.linear.short_conv",), "cutlass"),
        (
            (
                "attn_gym.linear.context_parallel",
                "attn_gym.linear.gdn.context_parallel",
                "attn_gym.linear.kda.context_parallel",
            ),
            "triton",
        ),
    ],
    ids=["short_conv", "context_parallel"],
)
def test_attention_modules_report_their_own_backend_at_build(
    broken_modules: tuple[str, ...], missing_backend: str
) -> None:
    """A module fails at build when an export it calls lacks its backend."""
    script = r"""
import importlib.abc
import importlib.util
import sys

broken_modules = sys.argv[1].split(",")
missing_backend = sys.argv[2]


class FakeAttentionGym(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # Every attn_gym module imports and exports a class under any name, except
    # the broken ones, whose backend is missing.

    def find_spec(self, fullname, path, target=None):
        if fullname in broken_modules:
            raise ModuleNotFoundError("blocked dependency", name=missing_backend)
        if fullname == "attn_gym" or fullname.startswith("attn_gym."):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def export(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {})

        module.__getattr__ = export


sys.meta_path.insert(0, FakeAttentionGym())

from torchtitan.models.common.attention import cp_gdn, cp_kda, gdn, kda
from torchtitan.models.common.attn_gym import is_missing_kernel

kda_kernel = kda.KDAKernel.Config()
gdn_kernel = gdn.GDNKernel.Config()
inner_kda = kda.InnerKDA.Config(head_dim=128, conv_kernel_size=4, kernel=kda_kernel)
inner_gdn = gdn.InnerGDN.Config(conv_kernel_size=4, kernel=gdn_kernel)
cp_inner_kda = cp_kda.ContextParallelInnerKDA.Config(
    head_dim=128, conv_kernel_size=4, kernel=kda_kernel
)
cp_inner_gdn = cp_gdn.ContextParallelInnerGDN.Config(
    conv_kernel_size=4, kernel=gdn_kernel
)


def backend_of(config):
    try:
        config.build()
    except ModuleNotFoundError as error:
        assert str(error) == "blocked dependency", error
        return error.name
    return None


# The kernels' own exports imported, so they build whichever module is broken.
assert backend_of(kda_kernel) is None
assert backend_of(gdn_kernel) is None

if "attn_gym.linear.short_conv" in broken_modules:
    # causal_conv1d is a placeholder: the inner modules that call it fail at
    # build, and so do their CP variants, which build on them.
    assert is_missing_kernel(kda.causal_conv1d)
    for config in (inner_kda, inner_gdn, cp_inner_kda, cp_inner_gdn):
        assert backend_of(config) == missing_backend, config
else:
    # Only the CP exports are placeholders: the inner modules build, and
    # their CP variants fail with the CP modules' backend.
    assert not is_missing_kernel(kda.causal_conv1d)
    assert is_missing_kernel(cp_kda.context_parallel_kda)
    assert backend_of(inner_kda) is None
    assert backend_of(inner_gdn) is None
    assert backend_of(cp_inner_kda) == missing_backend
    assert backend_of(cp_inner_gdn) == missing_backend
"""
    result = subprocess.run(
        [sys.executable, "-c", script, ",".join(broken_modules), missing_backend],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
