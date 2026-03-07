"""
Test minimal make_fx tracer against torchtitan models.

For each model we:
  1. Trace the full train step (forward + loss + autograd.grad) with make_fx
  2. Run 5 training steps, comparing loss and gradients from eager vs traced
  3. Verify bitwise equivalence at every step
"""

import contextlib
import io
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


# ---------------------------------------------------------------------------
# Subclass helpers
# ---------------------------------------------------------------------------

@dataclass
class SubclassMeta:
    cls: type
    attrs: list[str]
    ctx: Any
    inner_metas: dict[str, tuple[int, Any]]
    outer_size: torch.Size
    outer_stride: tuple[int, ...]


def unwrap_subclass(t: torch.Tensor) -> tuple[list[torch.Tensor], SubclassMeta | None]:
    if not is_traceable_wrapper_subclass(t):
        return [t], None
    attrs, ctx = t.__tensor_flatten__()
    all_inner = []
    inner_metas = {}
    for attr in attrs:
        inner_t = getattr(t, attr)
        tensors, meta = unwrap_subclass(inner_t)
        all_inner.extend(tensors)
        inner_metas[attr] = (len(tensors), meta)
    meta = SubclassMeta(
        cls=type(t),
        attrs=attrs,
        ctx=ctx,
        inner_metas=inner_metas,
        outer_size=t.size(),
        outer_stride=t.stride(),
    )
    return all_inner, meta


def wrap_to_subclass(plain_tensors: list[torch.Tensor], meta: SubclassMeta) -> torch.Tensor:
    inner_dict = {}
    idx = 0
    for attr in meta.attrs:
        num_inner, inner_meta = meta.inner_metas[attr]
        inner_tensors = plain_tensors[idx : idx + num_inner]
        idx += num_inner
        if inner_meta is None:
            inner_dict[attr] = inner_tensors[0]
        else:
            inner_dict[attr] = wrap_to_subclass(list(inner_tensors), inner_meta)
    return meta.cls.__tensor_unflatten__(inner_dict, meta.ctx, meta.outer_size, meta.outer_stride)


def wrap_inputs_to_subclasses(
    plain_args: tuple[torch.Tensor, ...],
    subclass_metas: list[tuple[int, SubclassMeta | None]],
) -> list[torch.Tensor]:
    wrapped = []
    idx = 0
    for num_tensors, meta in subclass_metas:
        tensors = plain_args[idx : idx + num_tensors]
        idx += num_tensors
        if meta is None:
            wrapped.append(tensors[0])
        else:
            wrapped.append(wrap_to_subclass(list(tensors), meta))
    return wrapped


def rewrap_outputs(outputs, output_subclass_metas):
    wrapped_outputs = []
    idx = 0
    for num_tensors, meta in output_subclass_metas:
        output_tensors = outputs[idx : idx + num_tensors]
        idx += num_tensors
        if meta is None:
            wrapped_outputs.append(output_tensors[0])
        else:
            wrapped_outputs.append(wrap_to_subclass(list(output_tensors), meta))
    return wrapped_outputs


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

def _remove_cpu_shadow_chains(gm: torch.fx.GraphModule) -> None:
    to_remove: set[torch.fx.Node] = set()

    for node in gm.graph.nodes:
        if node in to_remove:
            continue

        if not (
            node.op == "call_function"
            and node.target == torch.ops.aten.empty_strided.default
        ):
            continue
        device = node.kwargs.get("device")
        if device is None or device.type != "cpu":
            continue

        chain: set[torch.fx.Node] = set()
        queue = [node]
        feeds_gpu = False

        while queue and not feeds_gpu:
            current = queue.pop()
            if current in chain:
                continue
            chain.add(current)
            for user in current.users:
                val = user.meta.get("val")
                if isinstance(val, torch.Tensor) and val.device.type != "cpu":
                    if user.users:
                        feeds_gpu = True
                        break
                    chain.add(user)
                    continue
                queue.append(user)

        if not feeds_gpu:
            to_remove |= chain

    for node in reversed(list(gm.graph.nodes)):
        if node in to_remove:
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()


def trace_module(
    mod: nn.Module,
    args: tuple,
) -> torch.fx.GraphModule:
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {**named_parameters, **named_buffers}
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_len = len(params_and_buffers_flat)

    def functional_call(*all_args):
        flat_params = all_args[:params_len]
        user_args = all_args[params_len:]
        params = pytree.tree_unflatten(list(flat_params), params_spec)
        with stateless._reparametrize_module(mod, params):
            return mod.forward(*user_args)

    user_args_flat, user_args_spec = pytree.tree_flatten(args)
    full_args = tuple(params_and_buffers_flat) + tuple(user_args_flat)

    unwrapped_args = []
    subclass_metas = []

    for arg in full_args:
        if isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg):
            inner_tensors, meta = unwrap_subclass(arg)
            unwrapped_args.extend(inner_tensors)
            subclass_metas.append((len(inner_tensors), meta))
        else:
            unwrapped_args.append(arg)
            subclass_metas.append((1, None))

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def to_fake(t):
        if isinstance(t, torch.Tensor):
            return fake_mode.from_tensor(t)
        return t

    fake_args = tuple(to_fake(a) for a in unwrapped_args)

    output_subclass_metas = []

    def fn_with_subclass_handling(*plain_args):
        nonlocal output_subclass_metas
        output_subclass_metas = []

        wrapped_args = wrap_inputs_to_subclasses(plain_args, subclass_metas)

        params_args = wrapped_args[:params_len]
        user_args_wrapped = wrapped_args[params_len:]
        user_args_restored = pytree.tree_unflatten(
            list(user_args_wrapped), user_args_spec
        )

        outputs = functional_call(*params_args, *user_args_restored)

        flat_outputs, _ = pytree.tree_flatten(outputs)
        unwrapped_outputs = []
        for out in flat_outputs:
            if isinstance(out, torch.Tensor) and is_traceable_wrapper_subclass(out):
                inner, meta = unwrap_subclass(out)
                unwrapped_outputs.extend(inner)
                output_subclass_metas.append((len(inner), meta))
            else:
                unwrapped_outputs.append(out)
                output_subclass_metas.append((1, None))

        return unwrapped_outputs

    with fake_mode, preserve_node_meta():
        traced = make_fx(fn_with_subclass_handling, record_stack_traces=True)(
            *fake_args
        )

    _remove_cpu_shadow_chains(traced)

    traced._params_len = params_len
    traced._params_spec = params_spec
    traced._input_subclass_metas = subclass_metas
    traced._output_subclass_metas = output_subclass_metas

    return traced


def run_traced_module(
    traced: torch.fx.GraphModule,
    mod: nn.Module,
    args: tuple,
):
    params_and_buffers = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    params_flat, _ = pytree.tree_flatten(params_and_buffers)
    user_args_flat, _ = pytree.tree_flatten(args)

    all_args = []
    for a in itertools.chain(params_flat, user_args_flat):
        if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
            inner, _ = unwrap_subclass(a)
            all_args.extend(inner)
        else:
            all_args.append(a)

    return traced(*all_args)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.float32
BATCH_SIZE = 2
SEQ_LEN = 128
NUM_STEPS = 5
LR = 1e-3


def create_model(config_cls, model_config):
    model = config_cls(model_config)
    model.to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(DEVICE))
    return model


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


class TrainStepModule(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args):
        *fwd_args, labels = args
        logits = self.model(*fwd_args)
        loss = self.loss_fn(logits, labels)
        params = [p for _, p in self.model.named_parameters(remove_duplicate=False)]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)


def test_model(name, model_ref, model_traced_copy, tokens, labels, attn_masks=None, check_collective_ops=False):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    from contextlib import contextmanager
    from torch.nn.attention.flex_attention import flex_attention as raw_flex_attention
    from torchtitan.models.common.attention import FlexAttentionWrapper

    @contextmanager
    def _use_raw_flex_attn():
        original = FlexAttentionWrapper._compiled_flex_attn
        FlexAttentionWrapper._compiled_flex_attn = staticmethod(raw_flex_attention)
        try:
            yield
        finally:
            FlexAttentionWrapper._compiled_flex_attn = original

    fwd_args = (tokens,) if attn_masks is None else (tokens, attn_masks)

    train_step_ref = TrainStepModule(model_ref, get_loss)
    train_step_copy = TrainStepModule(model_traced_copy, get_loss)

    print(f"  Tracing train step (fwd + loss + bwd)...")
    try:
        with _use_raw_flex_attn():
            traced = trace_module(train_step_ref, (*fwd_args, labels))
    except Exception as e:
        print(f"  TRACE FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False

    print(f"  Trace succeeded. Graph has {len(list(traced.graph.nodes))} nodes.")

    if check_collective_ops:
        ag = sum(1 for n in traced.graph.nodes if 'all_gather_into_tensor' in str(n.target))
        rs = sum(1 for n in traced.graph.nodes if 'reduce_scatter_tensor' in str(n.target))
        print(f"  Collective ops: all_gather={ag}, reduce_scatter={rs}")
        assert ag > 0 and rs > 0, f"Expected collective ops in FSDP graph (ag={ag}, rs={rs})"

    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=LR)
    opt_copy = torch.optim.Adam(model_traced_copy.parameters(), lr=LR)

    all_passed = True
    for step in range(1, NUM_STEPS + 1):
        with _use_raw_flex_attn():
            logits_ref = model_ref(*fwd_args)
        loss_ref = get_loss(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]
        opt_ref.step()
        opt_ref.zero_grad()

        outputs = run_traced_module(traced, train_step_copy, (*fwd_args, labels))
        wrapped = rewrap_outputs(outputs, traced._output_subclass_metas)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]
        for p, g in zip(model_traced_copy.parameters(), grads_tr):
            p.grad = g
        opt_copy.step()
        opt_copy.zero_grad()

        loss_eq = torch.equal(loss_ref, loss_tr)
        grads_eq = all(
            torch.equal(gr, gt) for gr, gt in zip(grads_ref, grads_tr)
        )
        max_grad_diff = max(
            (gr - gt).abs().max().item() for gr, gt in zip(grads_ref, grads_tr)
        )

        passed = loss_eq and grads_eq
        status = "PASS" if passed else "FAIL"
        print(
            f"  Step {step}: {status}  "
            f"loss_eq={loss_eq}  grads_eq={grads_eq}  "
            f"max_grad_diff={max_grad_diff:.2e}  "
            f"loss_ref={loss_ref.item():.4f}  loss_tr={loss_tr.item():.4f}"
        )

        if not passed:
            all_passed = False

    if all_passed:
        print(f"  RESULT: ALL {NUM_STEPS} STEPS PASSED (bitwise equal loss + grads)")
    else:
        print(f"  RESULT: SOME STEPS FAILED")
    return all_passed


def _run_fsdp_test(name, config_cls, model_config, use_attn_masks=False):
    import torch.distributed as dist
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    try:
        world_size = dist.get_world_size()
        model_ref = create_model(config_cls, model_config)
        model_copy = create_model(config_cls, model_config)
        model_copy.load_state_dict(model_ref.state_dict())

        parallel_dims = ParallelDims(
            dp_shard=world_size, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1,
            world_size=world_size,
        )
        parallel_dims.build_mesh()
        fsdp_mesh = parallel_dims.get_mesh("fsdp")
        data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        data_parallel(model_copy, device_mesh=fsdp_mesh, mode="fully_shard")

        vocab_size = model_config.vocab_size
        tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

        attn_masks = None
        if use_attn_masks:
            from torchtitan.models.common.attention import create_attention_mask, get_causal_mask_mod
            attn_masks = create_attention_mask(get_causal_mask_mod(), 1, None, SEQ_LEN, SEQ_LEN)

        ctx = contextlib.redirect_stdout(io.StringIO()) if rank != 0 else contextlib.nullcontext()
        with ctx:
            return test_model(name, model_ref, model_copy, tokens, labels, attn_masks=attn_masks, check_collective_ops=True)
    finally:
        dist.destroy_process_group()


def _test_llama3_fsdp():
    from torchtitan.models.llama3 import Llama3Model, llama3_configs
    return _run_fsdp_test("llama3 debugmodel (fsdp)", Llama3Model, llama3_configs["debugmodel"])


def _test_qwen3_fsdp():
    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3 import qwen3_configs
    return _run_fsdp_test("qwen3 debugmodel (fsdp)", Qwen3Model, qwen3_configs["debugmodel"])


def _test_qwen3_moe_fsdp():
    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3 import qwen3_configs
    return _run_fsdp_test("qwen3 debugmodel_moe (fsdp)", Qwen3Model, qwen3_configs["debugmodel_moe"])


def _test_deepseek_v3_fsdp():
    from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
    from torchtitan.models.deepseek_v3 import deepseekv3_configs
    return _run_fsdp_test("deepseek_v3 debugmodel (fsdp)", DeepSeekV3Model, deepseekv3_configs["debugmodel"])


def _test_llama4_fsdp():
    from torchtitan.models.llama4.model import Llama4Model
    from torchtitan.models.llama4 import llama4_configs
    return _run_fsdp_test("llama4 debugmodel (fsdp)", Llama4Model, llama4_configs["debugmodel"], use_attn_masks=True)


def _test_gpt_oss_fsdp():
    import torch.distributed as dist
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel
    from torchtitan.models.gpt_oss.model import GptOssModel
    from torchtitan.models.gpt_oss import gptoss_configs
    from torchtitan.models.common.attention import (
        create_attention_mask, get_causal_mask_mod, get_sliding_window_mask_mod,
    )
    from torch.nn.attention.flex_attention import and_masks

    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    try:
        config = gptoss_configs["debugmodel"]
        world_size = dist.get_world_size()
        model_ref = create_model(GptOssModel, config)
        model_copy = create_model(GptOssModel, config)
        model_copy.load_state_dict(model_ref.state_dict())

        parallel_dims = ParallelDims(
            dp_shard=world_size, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1,
            world_size=world_size,
        )
        parallel_dims.build_mesh()
        fsdp_mesh = parallel_dims.get_mesh("fsdp")
        data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        data_parallel(model_copy, device_mesh=fsdp_mesh, mode="fully_shard")

        vocab_size = config.vocab_size
        tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        causal = get_causal_mask_mod()
        sw_size = config.layer.attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, SEQ_LEN, SEQ_LEN)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)), 1, None, SEQ_LEN, SEQ_LEN
        )
        attn_masks = {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}

        ctx = contextlib.redirect_stdout(io.StringIO()) if rank != 0 else contextlib.nullcontext()
        with ctx:
            return test_model("gpt_oss debugmodel (fsdp)", model_ref, model_copy, tokens, labels, attn_masks=attn_masks, check_collective_ops=True)
    finally:
        dist.destroy_process_group()


MODEL_REGISTRY = {
    "llama3": lambda: _test_llama3(),
    "qwen3": lambda: _test_qwen3(),
    "qwen3_moe": lambda: _test_qwen3_moe(),
    "deepseek_v3": lambda: _test_deepseek_v3(),
    "llama4": lambda: _test_llama4(),
    "gpt_oss": lambda: _test_gpt_oss(),
    "llama3_fsdp": lambda: _test_llama3_fsdp(),
    "qwen3_fsdp": lambda: _test_qwen3_fsdp(),
    "deepseek_v3_fsdp": lambda: _test_deepseek_v3_fsdp(),
    "llama4_fsdp": lambda: _test_llama4_fsdp(),
}


def _run_test(name, config_cls, model_config, use_attn_masks=False):
    vocab_size = model_config.vocab_size
    model_ref = create_model(config_cls, model_config)
    model_copy = create_model(config_cls, model_config)
    model_copy.load_state_dict(model_ref.state_dict())
    tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    if use_attn_masks:
        from torchtitan.models.common.attention import create_attention_mask, get_causal_mask_mod
        attn_masks = create_attention_mask(get_causal_mask_mod(), 1, None, SEQ_LEN, SEQ_LEN)
        return test_model(name, model_ref, model_copy, tokens, labels, attn_masks=attn_masks)

    return test_model(name, model_ref, model_copy, tokens, labels)


def _test_llama3():
    from torchtitan.models.llama3 import Llama3Model, llama3_configs
    return _run_test("llama3 debugmodel", Llama3Model, llama3_configs["debugmodel"])


def _test_qwen3():
    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3 import qwen3_configs
    return _run_test("qwen3 debugmodel", Qwen3Model, qwen3_configs["debugmodel"])


def _test_qwen3_moe():
    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3 import qwen3_configs
    return _run_test("qwen3 debugmodel_moe", Qwen3Model, qwen3_configs["debugmodel_moe"])


def _test_deepseek_v3():
    from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
    from torchtitan.models.deepseek_v3 import deepseekv3_configs
    return _run_test("deepseek_v3 debugmodel", DeepSeekV3Model, deepseekv3_configs["debugmodel"])


def _test_llama4():
    from torchtitan.models.llama4.model import Llama4Model
    from torchtitan.models.llama4 import llama4_configs
    return _run_test("llama4 debugmodel", Llama4Model, llama4_configs["debugmodel"], use_attn_masks=True)


def _test_gpt_oss():
    from torchtitan.models.gpt_oss.model import GptOssModel
    from torchtitan.models.gpt_oss import gptoss_configs
    from torchtitan.models.common.attention import (
        create_attention_mask, get_causal_mask_mod, get_sliding_window_mask_mod,
    )
    from torch.nn.attention.flex_attention import and_masks
    config = gptoss_configs["debugmodel"]
    vocab_size = config.vocab_size
    model_ref = create_model(GptOssModel, config)
    model_copy = create_model(GptOssModel, config)
    model_copy.load_state_dict(model_ref.state_dict())
    tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    causal = get_causal_mask_mod()
    sw_size = config.layer.attention.sliding_window_size
    basic_mask = create_attention_mask(causal, 1, None, SEQ_LEN, SEQ_LEN)
    sliding_window_mask = create_attention_mask(
        and_masks(causal, get_sliding_window_mask_mod(sw_size)), 1, None, SEQ_LEN, SEQ_LEN
    )
    attn_masks = {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}
    return test_model("gpt_oss debugmodel", model_ref, model_copy, tokens, labels, attn_masks=attn_masks)


def main():
    import subprocess
    models = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_REGISTRY.keys())

    if len(models) == 1 and models[0] in MODEL_REGISTRY:
        model_name = models[0]
        if model_name.endswith("_fsdp") and "LOCAL_RANK" not in os.environ:
            torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
            result = subprocess.run(
                [torchrun, "--nproc_per_node=8", __file__, model_name],
                capture_output=False,
                timeout=300,
            )
            return result.returncode
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        passed = MODEL_REGISTRY[model_name]()
        return 0 if passed else 1

    results = {}
    for model_name in models:
        print(f"\n--- Running {model_name} in subprocess ---")
        if model_name.endswith("_fsdp"):
            torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
            cmd = [torchrun, "--nproc_per_node=8", __file__, model_name]
        else:
            cmd = [sys.executable, __file__, model_name]
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=300,
        )
        results[model_name] = result.returncode == 0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {model_name}: {status}")

    if all(results.values()):
        print("\nAll models passed!")
        return 0
    else:
        print("\nSome models failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
