import os

import torch
from torchtitan.grpo.vllm_handling.vllm_patching.distributed_updater import (
    weight_updater_process,
)
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PatchedGPUModelRunner(GPUModelRunner):
    def load_model(self, eep_scale_up: bool = False) -> None:
        super().load_model(eep_scale_up)
        ctx = torch.multiprocessing.get_context("spawn")
        self.model.share_memory()
        lora_state_dict = {}
        if self.lora_config:
            self.lora_manager.add_adapter(
                LoRARequest(
                    lora_name="lora",
                    lora_int_id=1,
                    lora_path=os.environ.get("PEFT_PATH"),
                )
            )
            # self.lora_manager.set_active_adapters([1])
            lora_modules = self.lora_manager._adapter_manager.modules
            for key, value in lora_modules.items():
                if "experts" in key:
                    lora_state_dict[
                        key + ".w1_lora_a_stacked"
                    ] = value.w1_lora_a_stacked[0]
                    lora_state_dict[
                        key + ".w1_lora_b_stacked"
                    ] = value.w1_lora_b_stacked[0]
                    lora_state_dict[
                        key + ".w3_lora_a_stacked"
                    ] = value.w3_lora_a_stacked[0]
                    lora_state_dict[
                        key + ".w3_lora_b_stacked"
                    ] = value.w3_lora_b_stacked[0]
                    lora_state_dict[
                        key + ".w2_lora_a_stacked"
                    ] = value.w2_lora_a_stacked[0]
                    lora_state_dict[
                        key + ".w2_lora_b_stacked"
                    ] = value.w2_lora_b_stacked[0]
                elif "self_attn" in key:
                    if "o_proj" in key:
                        lora_state_dict[
                            key + ".o_proj.lora_a_stacked"
                        ] = value.lora_a_stacked[0]
                        lora_state_dict[
                            key + ".o_proj.lora_b_stacked"
                        ] = value.lora_b_stacked[0]
                    elif "qkv_proj" in key:
                        lora_state_dict[
                            key + ".q_proj.lora_a_stacked"
                        ] = value.lora_a_stacked[0]
                        lora_state_dict[
                            key + ".q_proj.lora_b_stacked"
                        ] = value.lora_b_stacked[0]
                        lora_state_dict[
                            key + ".k_proj.lora_a_stacked"
                        ] = value.lora_a_stacked[1]
                        lora_state_dict[
                            key + ".k_proj.lora_b_stacked"
                        ] = value.lora_b_stacked[1]
                        lora_state_dict[
                            key + ".v_proj.lora_a_stacked"
                        ] = value.lora_a_stacked[2]
                        lora_state_dict[
                            key + ".v_proj.lora_b_stacked"
                        ] = value.lora_b_stacked[2]
        state_dict = self.model.state_dict()
        state_dict.update(lora_state_dict)
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
