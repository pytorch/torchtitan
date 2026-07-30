# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-device forward parity and training smoke test for reduced Kimi K3.

This script compares a deterministic FP32 CPU reference against the same
TorchTitan model, weights, tokens, and image patches on another device. It then
runs real forward, backward, and AdamW updates on that device. Combine this test
with ``numerical_tests_kimi_k3.py``, which compares the CPU/CUDA TorchTitan
reference directly against the released HuggingFace implementation.

The optional ``--device_module`` argument imports a PyTorch out-of-tree device
extension before constructing the target device.
"""

import argparse
import importlib
import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle


_IMAGE_TOKEN_ID = 7


def _first_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Expected a tensor output, got {type(output).__name__}.")


def _capture_tensor(
    destination: dict[str, torch.Tensor],
    name: str,
    *,
    flatten: bool = False,
) -> Callable:
    def hook(_module, _inputs, output) -> None:
        value = _first_tensor(output).detach().float().cpu()
        if flatten:
            value = value.reshape(-1, value.shape[-1])
        destination[name] = value

    return hook


def _capture_router_ids(
    destination: dict[str, torch.Tensor],
    name: str,
) -> Callable:
    def hook(_module, _inputs, output) -> None:
        expert_ids = output[0]
        destination[name] = expert_ids.detach().reshape(-1, expert_ids.shape[-1]).cpu()

    return hook


def _register_hooks(
    model,
    outputs: dict[str, torch.Tensor],
) -> list[RemovableHandle]:
    handles: list[RemovableHandle] = []
    if model.vision_encoder is None:
        raise ValueError("The Kimi K3 debug config must include a vision encoder.")

    for layer_idx, layer in enumerate(model.vision_encoder.layers.values()):
        handles.append(
            layer.register_forward_hook(
                _capture_tensor(outputs, f"vision.layer.{layer_idx}", flatten=True)
            )
        )
    projector = model.vision_encoder.projector
    for name, module in (
        ("vision.projector.linear_1", projector.linear_1),
        ("vision.projector.activation", projector.activation),
        ("vision.projector.linear_2", projector.linear_2),
        ("vision.projector.post_norm", projector.post_norm),
    ):
        handles.append(
            module.register_forward_hook(_capture_tensor(outputs, name, flatten=True))
        )
    handles.append(
        projector.register_forward_hook(
            _capture_tensor(outputs, "vision.projected", flatten=True)
        )
    )

    for layer_idx, layer in enumerate(model.layers.values()):
        handles.append(
            layer.register_forward_hook(
                _capture_tensor(outputs, f"decoder.layer.{layer_idx}")
            )
        )
        attention = (
            layer.attention if layer.attention is not None else layer.delta_attention
        )
        assert attention is not None
        handles.append(
            attention.register_forward_hook(
                _capture_tensor(outputs, f"decoder.attention.{layer_idx}")
            )
        )
        if layer.moe is not None:
            handles.append(
                layer.moe.router.register_forward_hook(
                    _capture_router_ids(outputs, f"decoder.router_ids.{layer_idx}")
                )
            )
    return handles


def _compare_tensor(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    if reference.shape != actual.shape:
        raise AssertionError(
            f"{name}: reference shape {tuple(reference.shape)} does not match "
            f"device shape {tuple(actual.shape)}."
        )
    difference = (reference - actual).abs()
    cosine = F.cosine_similarity(
        reference.reshape(-1),
        actual.reshape(-1),
        dim=0,
    ).item()
    print(
        f"{name:32s} max={difference.max().item():.4e} "
        f"mean={difference.mean().item():.4e} cos={cosine:.8f}"
    )
    torch.testing.assert_close(
        actual,
        reference,
        atol=atol,
        rtol=rtol,
        msg=lambda message: f"{name} failed device parity:\n{message}",
    )


def _compare_captured(
    reference_outputs: dict[str, torch.Tensor],
    device_outputs: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    if reference_outputs.keys() != device_outputs.keys():
        raise AssertionError(
            "Captured output names differ: "
            f"reference-only={sorted(reference_outputs.keys() - device_outputs.keys())}, "
            f"device-only={sorted(device_outputs.keys() - reference_outputs.keys())}."
        )
    for name in reference_outputs:
        if ".router_ids." in name:
            if not torch.equal(reference_outputs[name], device_outputs[name]):
                mismatch = (
                    (reference_outputs[name] != device_outputs[name]).sum().item()
                )
                raise AssertionError(f"{name}: {mismatch} expert IDs differ.")
            print(f"{name:32s} exact expert IDs")
        else:
            _compare_tensor(
                name,
                reference_outputs[name],
                device_outputs[name],
                atol=atol,
                rtol=rtol,
            )


def _build_inputs(config, *, seed: int) -> dict[str, torch.Tensor | dict[str, int]]:
    vision_config = config.vision_encoder
    if vision_config is None:
        raise ValueError("The Kimi K3 debug config must include a vision encoder.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    patch_size = vision_config.patch_size
    grid_thw_N3 = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixel_values_NPK = torch.randn(
        1,
        16,
        3 * patch_size * patch_size,
        generator=generator,
        dtype=torch.float32,
    )
    tokens_BL = torch.tensor(
        [
            [
                11,
                23,
                _IMAGE_TOKEN_ID,
                _IMAGE_TOKEN_ID,
                _IMAGE_TOKEN_ID,
                _IMAGE_TOKEN_ID,
                17,
                31,
            ]
        ],
        dtype=torch.long,
    )
    return {
        "tokens": tokens_BL,
        "pixel_values": pixel_values_NPK,
        "grid_thw": grid_thw_N3,
        "special_tokens": {"image_id": _IMAGE_TOKEN_ID},
    }


def _move_inputs(
    inputs: dict[str, torch.Tensor | dict[str, int]],
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, int]]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def _synchronize(device: torch.device) -> None:
    device_api = getattr(torch, device.type, None)
    if device_api is not None and hasattr(device_api, "synchronize"):
        device_api.synchronize(device)


def _verify_device(device: torch.device) -> None:
    if device.type == "cpu":
        return
    device_api = getattr(torch, device.type, None)
    if device_api is None:
        raise RuntimeError(
            f"PyTorch has no registered {device.type!r} device module. "
            "Use --device_module to import its extension."
        )
    if hasattr(device_api, "is_available") and not device_api.is_available():
        raise RuntimeError(f"Requested device {device} is not available.")


@torch.no_grad()
def _run_forward_parity(
    reference_model,
    device_model,
    reference_inputs,
    device_inputs,
    *,
    atol: float,
    rtol: float,
) -> None:
    reference_outputs: dict[str, torch.Tensor] = {}
    device_outputs: dict[str, torch.Tensor] = {}
    reference_handles = _register_hooks(reference_model, reference_outputs)
    device_handles = _register_hooks(device_model, device_outputs)

    try:
        reference_logits_BLV = reference_model(**reference_inputs).float().cpu()
        device_logits_BLV = device_model(**device_inputs).float().cpu()
    finally:
        for handle in reference_handles + device_handles:
            handle.remove()

    _compare_captured(
        reference_outputs,
        device_outputs,
        atol=atol,
        rtol=rtol,
    )
    _compare_tensor(
        "multimodal.logits",
        reference_logits_BLV,
        device_logits_BLV,
        atol=atol,
        rtol=rtol,
    )


def _run_training(
    model,
    inputs,
    device: torch.device,
    *,
    train_dtype: torch.dtype,
    train_steps: int,
    learning_rate: float,
) -> None:
    if train_steps == 0:
        return

    model.to(dtype=train_dtype)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        foreach=False,
    )
    tokens_BL = inputs["tokens"]
    assert isinstance(tokens_BL, torch.Tensor)
    if model.lm_head is None:
        raise ValueError("Kimi K3 device training requires an LM head.")
    tracked_parameter = model.lm_head.weight
    tracked_before = tracked_parameter.detach().clone()

    for step_idx in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        start_time = time.perf_counter()
        logits_BLV = model(**inputs)
        loss = F.cross_entropy(
            logits_BLV[:, :-1].float().reshape(-1, logits_BLV.shape[-1]),
            tokens_BL[:, 1:].reshape(-1),
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=math.inf,
            foreach=False,
        )
        optimizer.step()
        _synchronize(device)
        elapsed = time.perf_counter() - start_time

        loss_value = loss.detach().float().cpu().item()
        grad_norm_value = grad_norm.detach().float().cpu().item()
        if not math.isfinite(loss_value) or not math.isfinite(grad_norm_value):
            raise AssertionError(
                f"Step {step_idx}: non-finite loss={loss_value} or "
                f"grad_norm={grad_norm_value}."
            )
        print(
            f"step={step_idx:02d} loss={loss_value:.8f} "
            f"grad_norm={grad_norm_value:.8f} elapsed={elapsed:.3f}s"
        )

    parameter_delta = (
        (tracked_parameter.detach().float() - tracked_before.float()).abs().max()
    )
    parameter_delta_value = parameter_delta.cpu().item()
    if parameter_delta_value == 0.0:
        raise AssertionError("AdamW completed without changing lm_head.weight.")
    print(f"lm_head.weight max update={parameter_delta_value:.4e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--device_module",
        help="Optional module that registers an out-of-tree PyTorch device.",
    )
    parser.add_argument(
        "--parity_dtype",
        default="float32",
        choices=("float32", "bfloat16"),
    )
    parser.add_argument(
        "--train_dtype",
        default="bfloat16",
        choices=("float32", "bfloat16"),
    )
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--train_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device_module is not None:
        importlib.import_module(args.device_module)

    from torchtitan.models.kimi_k3 import model_registry

    device = torch.device(args.device)
    _verify_device(device)
    parity_dtype = getattr(torch, args.parity_dtype)
    train_dtype = getattr(torch, args.train_dtype)
    if args.train_steps < 0:
        raise ValueError("--train_steps must be non-negative.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    config = model_registry("debugmodel").model
    reference_model = config.build()
    reference_model.init_states()
    device_model = config.build()
    device_model.init_states()
    device_model.load_state_dict(reference_model.state_dict(), strict=True)

    reference_model = reference_model.to(dtype=torch.float32).eval()
    device_model = device_model.to(device=device, dtype=parity_dtype).eval()
    reference_inputs = _build_inputs(config, seed=args.seed)
    device_inputs = _move_inputs(reference_inputs, device)

    print(f"\nForward parity: cpu/float32 -> {device}/{args.parity_dtype}")
    _run_forward_parity(
        reference_model,
        device_model,
        reference_inputs,
        device_inputs,
        atol=args.atol,
        rtol=args.rtol,
    )

    del reference_model
    print(f"\nTraining smoke: {device}/{args.train_dtype}, steps={args.train_steps}")
    _run_training(
        device_model,
        device_inputs,
        device,
        train_dtype=train_dtype,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
    )
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
