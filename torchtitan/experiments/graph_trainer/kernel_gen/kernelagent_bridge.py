"""Bridge to KernelAgent for Triton kernel generation and optimization.

Requires KernelAgent to be available at ~/local/KernelAgent or at the path
specified by the KERNEL_AGENT_ROOT environment variable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_META_CA_BUNDLE = "/etc/pki/tls/certs/fb_certs.pem"


def _ensure_api_key() -> None:
    """Ensure ANTHROPIC_API_KEY is set, fetching it via claude-meta if needed."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        key = subprocess.check_output(
            ["claude-meta", "inference", "get-secret", "OPUS_FAST_API_KEY"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            logger.info("Set ANTHROPIC_API_KEY via claude-meta")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def _ensure_proxy() -> None:
    """Configure proxy and SSL for Meta corporate environment.

    Sets proxy env vars via ``with-proxy`` and patches the ``anthropic``
    SDK's httpx client to trust the Meta CA bundle so requests can go
    through the corporate forward proxy.
    """
    if not os.environ.get("HTTPS_PROXY"):
        try:
            result = subprocess.run(
                ["with-proxy", "env"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.lower() in ("http_proxy", "https_proxy", "no_proxy"):
                            os.environ.setdefault(k, v)
                            os.environ.setdefault(k.upper(), v)
                logger.info("Configured Meta proxy via with-proxy")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if os.path.exists(_META_CA_BUNDLE):
            os.environ.setdefault("SSL_CERT_FILE", _META_CA_BUNDLE)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", _META_CA_BUNDLE)

    # Always patch — subprocesses inherit env vars but not in-memory patches
    _patch_anthropic_ssl()


_ANTHROPIC_PUBLIC_BASE_URL = "https://api.anthropic.com"


def _patch_anthropic_ssl() -> None:
    """Patch ``anthropic.Anthropic`` to go through the corporate forward
    proxy to ``api.anthropic.com`` instead of the internal AI gateway.

    The installed ``anthropic`` SDK auto-discovers the Meta AI gateway
    (``anthropic.ai-gateway.fbinfra.net``) which requires mTLS that
    standalone scripts don't have.  We override ``base_url`` to the
    public API endpoint and route through ``fwdproxy`` with
    ``verify=False`` (the proxy re-signs responses).
    """
    try:
        import anthropic
        import httpx as _httpx
    except ImportError:
        return

    if getattr(anthropic.Anthropic, "_meta_ssl_patched", False):
        return

    _orig_init = anthropic.Anthropic.__init__

    def _patched_init(self: object, *args: object, **kwargs: object) -> None:
        if "base_url" not in kwargs:
            kwargs["base_url"] = _ANTHROPIC_PUBLIC_BASE_URL
        if "http_client" not in kwargs:
            kwargs["http_client"] = _httpx.Client(verify=False)
        _orig_init(self, *args, **kwargs)

    anthropic.Anthropic.__init__ = _patched_init  # type: ignore[assignment]
    anthropic.Anthropic._meta_ssl_patched = True  # type: ignore[attr-defined]
    logger.info("Patched anthropic.Anthropic for Meta proxy")


def _ensure_kernel_agent_on_path() -> Path:
    root = Path(
        os.environ.get("KERNEL_AGENT_ROOT", os.path.expanduser("~/local/KernelAgent"))
    )
    if not root.exists():
        raise RuntimeError(
            f"KernelAgent not found at {root}. "
            "Set KERNEL_AGENT_ROOT to the correct path."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def generate_kernel(
    problem_description: str,
    *,
    num_workers: int = 4,
    max_rounds: int = 10,
    model_name: str | None = None,
    output_dir: str | None = None,
    test_code: str | None = None,
) -> dict[str, Any]:
    """Generate a Triton kernel from a problem description using KernelAgent.

    Args:
        problem_description: KernelBench-style problem string containing a
            Model class, get_inputs(), and get_init_inputs().
        num_workers: Number of parallel generation workers.
        max_rounds: Max LLM refinement rounds per worker.
        model_name: LLM model name (default: from env or claude-opus).
        output_dir: Directory to save generation logs.
        test_code: Optional additional test code for validation.

    Returns:
        dict with keys:
            success: bool
            kernel_code: str (if success)
            session_dir: str
            message: str (if failure)
    """
    _ensure_api_key()
    _ensure_kernel_agent_on_path()
    _ensure_proxy()
    from triton_kernel_agent import TritonKernelAgent

    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "max_rounds": max_rounds,
    }
    if model_name is not None:
        kwargs["model_name"] = model_name
    if output_dir is not None:
        kwargs["log_dir"] = output_dir

    agent = TritonKernelAgent(**kwargs)
    try:
        return agent.generate_kernel(
            problem_description,
            test_code=test_code,
        )
    finally:
        agent.cleanup()


_DEFAULT_MODEL = "claude-opus-4-6"

# GPU names recognized by KernelAgent's spec database.
_GPU_NAME_MAP = {
    "NVIDIA H100": "NVIDIA H100 NVL 94GB",
    "NVIDIA H100 80GB HBM3": "NVIDIA H100 SXM5 80GB",
    "NVIDIA A100-SXM4-80GB": "NVIDIA A100 SXM4 80GB",
    "NVIDIA A100-SXM4-40GB": "NVIDIA A100 SXM4 40GB",
    "NVIDIA A100-PCIE-80GB": "NVIDIA A100 PCIe 80GB",
    "NVIDIA A100-PCIE-40GB": "NVIDIA A100 PCIe 40GB",
}


def _detect_gpu_name() -> str:
    """Auto-detect GPU and map to a name KernelAgent recognizes."""
    import torch

    raw = torch.cuda.get_device_name(0)
    if raw in _GPU_NAME_MAP:
        return _GPU_NAME_MAP[raw]
    # Try substring match
    for key, val in _GPU_NAME_MAP.items():
        if key in raw or raw in key:
            return val
    logger.warning(f"Unknown GPU '{raw}', falling back to '{raw}'")
    return raw


def _clean_problem_source(path: Path) -> Path:
    """Return a path to a valid-Python problem file.

    problem.py files may have a description preamble before ``import torch``.
    The optimizer exec's the file directly, so we write a clean copy if needed.
    """
    content = path.read_text()
    idx = content.find("import torch")
    if idx <= 0:
        return path
    clean_path = path.with_name("problem_clean.py")
    clean_path.write_text(content[idx:])
    return clean_path


def _find_test_code(problem_dir: Path) -> str | None:
    """Find the latest auto-generated test from the generation session logs."""
    sessions = sorted(problem_dir.glob("logs/session_*/test_0.py"))
    if sessions:
        return sessions[-1].read_text()
    return None


_SYNTH_TEST_TEMPLATE = '''\
"""Synthesized correctness test: triton kernel vs eager Model."""
import torch
from problem import Model, get_inputs


def _close(a, b, atol=8e-2, rtol=1e-2):
    a = a.float() if isinstance(a, torch.Tensor) else a
    b = b.float() if isinstance(b, torch.Tensor) else b
    diff = (a - b).abs()
    budget = atol + rtol * b.abs()
    return (diff <= budget).all().item(), float(diff.max())


def test_kernel():
    from kernel import kernel_function
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    inputs = get_inputs()
    model = Model()
    with torch.no_grad():
        expected = model(*inputs)
        result = kernel_function(*inputs)

    if not isinstance(expected, (tuple, list)):
        expected = (expected,)
        if isinstance(result, torch.Tensor):
            result = (result,)
        elif isinstance(result, tuple) and len(result) == 1:
            pass
    if len(expected) != len(result):
        print(f"FAIL: tuple length mismatch {len(expected)} vs {len(result)}")
        return False
    for i, (e, r) in enumerate(zip(expected, result)):
        if not isinstance(e, torch.Tensor) or not isinstance(r, torch.Tensor):
            continue
        if e.shape != r.shape:
            print(f"FAIL: output[{i}] shape {e.shape} vs {r.shape}")
            return False
        ok, max_abs = _close(r, e)
        if not ok:
            print(f"FAIL: output[{i}] max_abs={max_abs:.4e}")
            return False
    print("PASS")
    return True


if __name__ == "__main__":
    test_kernel()
'''


def _make_test_code(problem_path: Path) -> str:
    """Synthesize a minimal test_code that compares kernel_function to Model.

    Used when no auto-generated test_0.py exists (e.g. problems extracted
    from a live FX graph rather than generated via TritonKernelAgent).
    """
    return _SYNTH_TEST_TEMPLATE


def optimize_kernel(
    kernel_code: str,
    problem_path: str | Path,
    test_code: str | None = None,
    *,
    strategy: str = "beam_search",
    num_workers: int = 4,
    max_rounds: int = 10,
    model_name: str | None = None,
    output_dir: str | None = None,
    gpu_name: str | None = None,
) -> dict[str, Any]:
    """Optimize an existing Triton kernel using KernelAgent's NCU-guided optimizer.

    Args:
        kernel_code: The initial Triton kernel code (must define kernel_function).
        problem_path: Path to problem.py defining the reference Model.
        test_code: Correctness test code string. If None, auto-detected from
            the generation session logs.
        strategy: Optimization strategy ("beam_search" or "greedy").
        num_workers: Number of parallel optimization workers.
        max_rounds: Maximum optimization rounds.
        model_name: LLM model name (default: claude-opus).
        output_dir: Directory for optimization logs.
        gpu_name: GPU name for NCU analysis (auto-detected if None).

    Returns:
        dict with keys:
            success: bool
            kernel_code: str (best optimized kernel)
            best_time_ms: float
            pytorch_baseline_ms: float
            initial_kernel_time_ms: float
    """
    _ensure_api_key()
    _ensure_kernel_agent_on_path()
    _ensure_proxy()
    from triton_kernel_agent.opt_manager import OptimizationManager

    problem_path = Path(problem_path)
    clean_problem = _clean_problem_source(problem_path)

    if test_code is None:
        test_code = _find_test_code(problem_path.parent)
    if test_code is None:
        test_code = _make_test_code(problem_path)

    if gpu_name is None:
        gpu_name = _detect_gpu_name()

    kwargs: dict[str, Any] = {
        "strategy": strategy,
        "num_workers": num_workers,
        "max_rounds": max_rounds,
        "openai_model": model_name or _DEFAULT_MODEL,
        "gpu_name": gpu_name,
    }
    if output_dir is not None:
        kwargs["log_dir"] = output_dir

    manager = OptimizationManager(**kwargs)
    return manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=clean_problem,
        test_code=test_code,
        max_rounds=max_rounds,
    )


def save_problem_files(
    problem_description: str,
    output_dir: str | Path,
    *,
    kernel_code: str | None = None,
    test_code: str | None = None,
) -> Path:
    """Save problem description (and optionally kernel/test) to files on disk.

    Creates the directory structure expected by KernelAgent's optimizer:
        output_dir/problem.py
        output_dir/input.py    (if kernel_code provided)
        output_dir/test.py     (if test_code provided)

    Returns the output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "problem.py").write_text(problem_description)
    if kernel_code is not None:
        (out / "input.py").write_text(kernel_code)
    if test_code is not None:
        (out / "test.py").write_text(test_code)

    return out
