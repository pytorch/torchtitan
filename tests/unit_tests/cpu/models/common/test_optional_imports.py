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
        assert "attn-gym[linear,cudnn]==0.0.17" in str(error)
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
        assert "attn-gym[linear,cudnn]==0.0.17" in str(error)
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


# Each case breaks one attn_gym module, or one export a module resolves on
# first access (attn_gym.linear._lazy), on a backend of its own, and names the
# modules whose build must then fail with it. Breaking one export at a time
# makes every entry of the require_attn_gym lists load-bearing: drop an export
# from a list and its module builds in the case that breaks that export.
_BUILD_FAILURE_CASES = [
    ("attn_gym.linear.short_conv", "inner_kda,inner_gdn,cp_inner_kda,cp_inner_gdn"),
    ("causal_conv1d", "inner_kda,inner_gdn,cp_inner_kda,cp_inner_gdn"),
    ("chunk_kda", "kda_kernel,inner_kda,cp_inner_kda"),
    ("bound_gate", "kda_kernel,inner_kda,cp_inner_kda"),
    (
        "l2norm",
        "kda_kernel,gdn_kernel,inner_kda,inner_gdn,cp_inner_kda,cp_inner_gdn",
    ),
    ("chunk_gdn", "gdn_kernel,inner_gdn,cp_inner_gdn"),
    ("recurrent_gdn", "gdn_kernel,inner_gdn,cp_inner_gdn"),
    ("context_parallel_kda", "cp_inner_kda"),
    ("context_parallel_gdn", "cp_inner_gdn"),
    ("context_parallel_conv_history", "cp_inner_kda,cp_inner_gdn"),
    ("ContextParallelRouting", "cp_inner_kda,cp_inner_gdn"),
]


@pytest.mark.parametrize(
    "broken, failing",
    _BUILD_FAILURE_CASES,
    ids=[broken for broken, _ in _BUILD_FAILURE_CASES],
)
def test_attention_modules_report_their_own_backend_at_build(
    broken: str, failing: str
) -> None:
    """A module fails at build when an export it calls lacks its backend."""
    script = r"""
import importlib.abc
import importlib.util
import sys

broken = sys.argv[1]
failing = set(sys.argv[2].split(","))
missing_backend = broken.rsplit(".", 1)[-1] + "_backend"


def blocked():
    return ModuleNotFoundError("blocked dependency", name=missing_backend)


class FakeAttentionGym(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # Every attn_gym module imports and exports a class under any name. The
    # broken module fails to import on its backend. The broken export fails on
    # its backend when first accessed, as attn_gym.linear._lazy does, with the
    # backend's error chained to an ImportError.

    def find_spec(self, fullname, path, target=None):
        if fullname == broken:
            raise blocked()
        if fullname == "attn_gym" or fullname.startswith("attn_gym."):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def export(name):
            if name.startswith("__"):
                raise AttributeError(name)
            if name == broken:
                raise ImportError(f"{name} needs {missing_backend}") from blocked()
            return type(name, (), {})

        module.__getattr__ = export


sys.meta_path.insert(0, FakeAttentionGym())

import torchtitan.models.common.attn_gym as facade
from torchtitan.models.common.attention import cp_gdn, cp_kda, gdn, kda

# The placeholder is named after what failed: the module, or the export.
if broken in facade._EXPORTS:
    export, placeholder_name = broken, f"{facade._EXPORTS[broken]}.{broken}"
else:
    export = next(name for name, module in facade._EXPORTS.items() if module == broken)
    placeholder_name = broken
placeholder = getattr(facade, export)
assert facade.is_missing_kernel(placeholder), placeholder
assert placeholder.__name__ == f"_MissingKernel[{placeholder_name}]", placeholder

kda_kernel = kda.KDAKernel.Config()
gdn_kernel = gdn.GDNKernel.Config()
configs = {
    "kda_kernel": kda_kernel,
    "gdn_kernel": gdn_kernel,
    "inner_kda": kda.InnerKDA.Config(
        head_dim=128, conv_kernel_size=4, kernel=kda_kernel
    ),
    "inner_gdn": gdn.InnerGDN.Config(conv_kernel_size=4, kernel=gdn_kernel),
    "cp_inner_kda": cp_kda.ContextParallelInnerKDA.Config(
        head_dim=128, conv_kernel_size=4, kernel=kda_kernel
    ),
    "cp_inner_gdn": cp_gdn.ContextParallelInnerGDN.Config(
        conv_kernel_size=4, kernel=gdn_kernel
    ),
}
assert failing <= set(configs), failing


# A broken export keeps attn_gym's message, which names the backend; a broken
# module reports the backend's own error.
expected_message = (
    f"{broken} needs {missing_backend}"
    if broken in facade._EXPORTS
    else "blocked dependency"
)


def backend_of(config):
    try:
        config.build()
    except ModuleNotFoundError as error:
        assert str(error) == expected_message, error
        return error.name
    return None


# A module fails with the broken backend if and only if it, or a module it
# builds, lists the broken export; every other module builds.
for name, config in configs.items():
    backend = backend_of(config)
    expected = missing_backend if name in failing else None
    assert backend == expected, (name, backend, expected)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, broken, failing],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_attention_gym_lazy_exports_with_missing_backend() -> None:
    """Exports attn_gym resolves on first access defer like a failing module."""
    script = r"""
import importlib
from types import ModuleType
from unittest.mock import Mock

import spmd_types as spmd
import torchtitan.models.common.attn_gym as facade
from torchtitan.models.common.attention import gdn


def lazy_export(name):
    # attn_gym.linear._lazy imports the owning module on first access and wraps
    # its error in an ImportError chained from the backend's ModuleNotFoundError.
    if name == "causal_conv1d":
        cause = ModuleNotFoundError("No module named 'cutlass'", name="cutlass")
    elif name == "causal_conv1d_decode":
        cause = ModuleNotFoundError(
            "No module named 'attn_gym.linear.short_conv.cute'",
            name="attn_gym.linear.short_conv.cute",
        )
    elif name == "paged_causal_conv1d":
        raise ImportError("cannot import name 'paged_causal_conv1d'")
    else:
        raise AttributeError(name)
    raise ImportError(f"{name} requires the optional CuTeDSL backend") from cause


short_conv = ModuleType("attn_gym.linear.short_conv")
short_conv.__getattr__ = lazy_export
available = {
    "attn_gym": ModuleType("attn_gym"),
    "attn_gym.linear.short_conv": short_conv,
}


def load_module(name):
    if name in available:
        return available[name]
    raise ModuleNotFoundError("blocked dependency", name="triton")


facade.import_module = load_module
facade._BACKEND_ERRORS.clear()
facade._BACKEND_KERNELS.clear()
for name in facade._EXPORTS:
    facade.__dict__.pop(name, None)
spmd.register_local_autograd_function = Mock()

importlib.reload(gdn)

assert gdn.causal_conv1d is facade.causal_conv1d
assert facade.is_missing_kernel(facade.causal_conv1d)
assert facade.causal_conv1d is not facade._MissingKernel
recorded = facade._BACKEND_ERRORS["attn_gym.linear.short_conv.causal_conv1d"]
assert recorded.name == "cutlass", recorded
# attn_gym's message is kept, and the backend's own error stays in the chain.
assert str(recorded) == "causal_conv1d requires the optional CuTeDSL backend", recorded
assert str(recorded.__cause__.__cause__) == "No module named 'cutlass'", recorded
for use in (facade.causal_conv1d, lambda: facade.require_attn_gym(facade.causal_conv1d)):
    try:
        use()
    except ModuleNotFoundError as error:
        assert error.name == "cutlass", error
        assert str(error) == "causal_conv1d requires the optional CuTeDSL backend", error
    else:
        raise AssertionError("A kernel with a missing backend ran")

# A missing attn_gym module, or an ImportError that is not a missing module,
# is breakage and stays loud.
for name in ("causal_conv1d_decode", "paged_causal_conv1d"):
    try:
        getattr(facade, name)
    except ImportError as error:
        assert not isinstance(error, ModuleNotFoundError), error
        assert name not in facade.__dict__
    else:
        raise AssertionError(f"{name} got a placeholder for an attn_gym error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
