import dataclasses
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.parallelize import parallelize_qwen3


# Scheduled-token count for the single forward. The unpadded path fails when
# L < TP and passes when L >= TP.
L = int(os.environ.get("REPRO_L", "1"))
PAD_TO_TP = os.environ.get("REPRO_PAD_TO_TP", "0") == "1"


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.set_default_dtype(torch.bfloat16)
    world_size = dist.get_world_size()

    parallelism = ParallelismConfig(
        tensor_parallel_degree=4,
        expert_parallel_degree=4,
        enable_sequence_parallel=False,
    )

    config = model_registry("debugmodel_moe").model
    # Swap attention to SDPA, matching the vLLM wrapper.
    config = dataclasses.replace(
        config,
        layers=[
            dataclasses.replace(
                layer,
                attention=dataclasses.replace(
                    layer.attention,
                    inner_attention=ScaledDotProductAttention.Config(),
                ),
            )
            for layer in config.layers
        ],
    )

    @dataclass(kw_only=True, slots=True)
    class _InferenceConfig:
        parallelism: ParallelismConfig

    config.update_from_config(config=_InferenceConfig(parallelism=parallelism))

    parallel_dims = ParallelDims.from_config(parallelism, world_size=world_size)
    with torch.device("meta"):
        model = config.build()
    model = parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        training=TrainingConfig(),
        parallelism=parallelism,
        compile_config=CompileConfig(enable=False),
        ac_config=None,
        dump_folder="",
        skip_dp=True,
    )
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    model.eval()

    # The padded mode mirrors TorchTitanGPUModelRunner: the model sees a TP-
    # divisible extent while num_actual_tokens retains the scheduled count.
    with torch.no_grad():
        tp_size = parallelism.tensor_parallel_degree
        model_L = ((L + tp_size - 1) // tp_size) * tp_size if PAD_TO_TP else L
        tokens_2d = torch.randint(
            0, config.vocab_size, (model_L,), device="cuda"
        ).unsqueeze(0)
        positions = torch.arange(model_L, device="cuda").unsqueeze(0)
        num_actual_tokens = (
            torch.tensor([L], dtype=torch.int64, device="cuda")
            if PAD_TO_TP
            else None
        )
        h = model.tok_embeddings(tokens_2d)
        for layer in model.layers.values():
            h = layer(
                h,
                attention_masks=None,
                positions=positions,
                num_actual_tokens=num_actual_tokens,
            )
        model.norm(h)

    if dist.get_rank() == 0:
        print(
            f"forward OK for actual L={L}, model L={model_L}, "
            f"pad_to_tp={PAD_TO_TP}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
