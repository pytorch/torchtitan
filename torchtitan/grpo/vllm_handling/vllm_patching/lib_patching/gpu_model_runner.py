import torch
from torchtitan.grpo.vllm_handling.vllm_patching.distributed_updater import (
    weight_updater_process,
)
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PatchedGPUModelRunner(GPUModelRunner):
    def load_model(self, eep_scale_up: bool = False) -> None:
        super().load_model(eep_scale_up)
        ctx = torch.multiprocessing.get_context("spawn")
        self.model.share_memory()
        state_dict = self.model.state_dict()
        for key, val in state_dict.items():
            val.share_memory_()
        num_heads = getattr(self.model_config.hf_text_config, "num_attention_heads", 0)
        num_kv_heads = self.model_config.get_total_num_kv_heads()
        tp_rank = get_tensor_model_parallel_rank()
        gpu_id = torch.cuda.device(self.device).idx
        self.mdl_distributed_proc = ctx.Process(
            target=weight_updater_process,
            args=(
                state_dict,
                num_heads,
                num_kv_heads,
                tp_rank,
                self.parallel_config.tensor_parallel_size,
                gpu_id,
            ),
            daemon=True,
        )
        self.mdl_distributed_proc.start()
