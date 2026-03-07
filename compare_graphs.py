import contextlib
import io
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention as raw_flex_attention

from test_tracer import (
    TrainStepModule,
    create_model,
    get_loss,
    trace_module,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.compiler_toolkit.common_utils import register_blockmask_pytree_node
from torchtitan.experiments.compiler_toolkit.graph_utils import export_joint
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel
from torchtitan.models.common.attention import FlexAttentionWrapper

DEVICE = "cuda"
DTYPE = torch.float32
BATCH_SIZE = 2
SEQ_LEN = 128
OUTPUT_DIR = Path("graphs")


class ForwardLossModule(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args):
        *fwd_args, labels = args
        logits = self.model(*fwd_args)
        return self.loss_fn(logits, labels)


def _setup_distributed():
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    parallel_dims = ParallelDims(
        dp_shard=world_size, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    return rank, fsdp_mesh


def _readable(gm):
    return gm.print_readable(print_output=False, include_stride=True, include_device=True)


def _op_freq(gm):
    freq = Counter()
    for node in gm.graph.nodes:
        if node.op == "call_function":
            freq[str(node.target)] += 1
    return freq


def _diff_summary(freq_a, freq_b):
    all_ops = sorted(set(freq_a) | set(freq_b))
    lines = []
    for op in all_ops:
        a, b = freq_a.get(op, 0), freq_b.get(op, 0)
        if a != b:
            lines.append(f"  {op}: non_strict={a}  titan={b}  delta={b - a:+d}")
    return lines


def _use_raw_flex_attn():
    @contextlib.contextmanager
    def ctx():
        original = FlexAttentionWrapper._compiled_flex_attn
        FlexAttentionWrapper._compiled_flex_attn = staticmethod(raw_flex_attention)
        try:
            yield
        finally:
            FlexAttentionWrapper._compiled_flex_attn = original
    return ctx()


def _capture_non_strict(model, fwd_args, labels):
    wrapper = TrainStepModule(model, get_loss)
    with _use_raw_flex_attn():
        gm = trace_module(wrapper, (*fwd_args, labels))
    return gm


def _capture_titan(model, fwd_args, labels):
    wrapper = ForwardLossModule(model, get_loss)
    args = (*fwd_args, labels)
    joint_with_desc, _ = export_joint(wrapper, args)
    return joint_with_desc.graph_module


def _make_inputs(vocab_size, attn_masks=None):
    tokens = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    if attn_masks is not None:
        if isinstance(attn_masks, dict):
            return (tokens, attn_masks), labels
        else:
            return (tokens, attn_masks), labels
    return (tokens,), labels


def _run_model(name, config_cls, model_config, fsdp_mesh, rank, use_attn_masks=False, gpt_oss_masks=False):
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model_ns = create_model(config_cls, model_config)
    torch.manual_seed(42)
    model_titan = create_model(config_cls, model_config)

    data_parallel(model_ns, device_mesh=fsdp_mesh, mode="fully_shard")
    data_parallel(model_titan, device_mesh=fsdp_mesh, mode="fully_shard")

    attn_masks = None
    if gpt_oss_masks or use_attn_masks:
        register_blockmask_pytree_node()
    if gpt_oss_masks:
        from torchtitan.models.common.attention import (
            create_attention_mask, get_causal_mask_mod, get_sliding_window_mask_mod,
        )
        from torch.nn.attention.flex_attention import and_masks
        causal = get_causal_mask_mod()
        sw_size = model_config.layer.attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, SEQ_LEN, SEQ_LEN)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)), 1, None, SEQ_LEN, SEQ_LEN
        )
        attn_masks = {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}
    elif use_attn_masks:
        from torchtitan.models.common.attention import create_attention_mask, get_causal_mask_mod
        attn_masks = create_attention_mask(get_causal_mask_mod(), 1, None, SEQ_LEN, SEQ_LEN)

    fwd_args, labels = _make_inputs(model_config.vocab_size, attn_masks)

    print(f"  Capturing non_strict graph...")
    ns_text, ns_freq, ns_nodes = None, None, None
    try:
        gm_ns = _capture_non_strict(model_ns, fwd_args, labels)
        ns_text = _readable(gm_ns)
        ns_nodes = len(list(gm_ns.graph.nodes))
        ns_freq = _op_freq(gm_ns)
        print(f"  non_strict: {ns_nodes} nodes")
    except Exception as e:
        print(f"  non_strict FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print(f"  Capturing titan graph...")
    titan_text, titan_freq, titan_nodes = None, None, None
    try:
        gm_titan = _capture_titan(model_titan, fwd_args, labels)
        titan_text = _readable(gm_titan)
        titan_nodes = len(list(gm_titan.graph.nodes))
        titan_freq = _op_freq(gm_titan)
        print(f"  titan: {titan_nodes} nodes")
    except Exception as e:
        print(f"  titan FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if ns_text is not None:
            (OUTPUT_DIR / f"non_strict_{name}.txt").write_text(ns_text)
        if titan_text is not None:
            (OUTPUT_DIR / f"titan_{name}.txt").write_text(titan_text)

        if ns_freq is not None and titan_freq is not None:
            diff_lines = _diff_summary(ns_freq, titan_freq)
            summary = []
            summary.append(f"Model: {name}")
            summary.append(f"non_strict nodes: {ns_nodes}")
            summary.append(f"titan nodes: {titan_nodes}")
            summary.append(f"Op frequency differences ({len(diff_lines)} ops differ):")
            summary.extend(diff_lines)
            summary_text = "\n".join(summary)
            print(summary_text)
            (OUTPUT_DIR / f"diff_{name}.txt").write_text(summary_text)


def main():
    rank, fsdp_mesh = _setup_distributed()

    models_arg = sys.argv[1:] if len(sys.argv) > 1 else None

    registry = {}

    from torchtitan.models.llama3 import Llama3Model, llama3_configs
    registry["llama3"] = (Llama3Model, llama3_configs["debugmodel"], False, False)

    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3 import qwen3_configs
    registry["qwen3"] = (Qwen3Model, qwen3_configs["debugmodel"], False, False)

    from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
    from torchtitan.models.deepseek_v3 import deepseekv3_configs
    registry["deepseek_v3"] = (DeepSeekV3Model, deepseekv3_configs["debugmodel"], False, False)

    from torchtitan.models.llama4.model import Llama4Model
    from torchtitan.models.llama4 import llama4_configs
    registry["llama4"] = (Llama4Model, llama4_configs["debugmodel"], True, False)

    from torchtitan.models.gpt_oss.model import GptOssModel
    from torchtitan.models.gpt_oss import gptoss_configs
    registry["gpt_oss"] = (GptOssModel, gptoss_configs["debugmodel"], False, True)

    models = models_arg if models_arg else list(registry.keys())

    ctx = contextlib.redirect_stdout(io.StringIO()) if rank != 0 else contextlib.nullcontext()
    with ctx:
        for name in models:
            if name not in registry:
                print(f"Unknown model: {name}, skipping")
                continue
            config_cls, model_config, use_attn_masks, gpt_oss_masks = registry[name]
            _run_model(name, config_cls, model_config, fsdp_mesh, rank,
                       use_attn_masks=use_attn_masks, gpt_oss_masks=gpt_oss_masks)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
