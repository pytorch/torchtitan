import atexit
import multiprocessing as _mp
import os

import torch
from torchtitan.grpo.vllm_handling.vllm_patching.distributed_updater import (
    weight_updater_process,
)
from vllm.distributed import get_ep_group, get_pp_group, get_tensor_model_parallel_rank
from vllm.lora.request import LoRARequest

# TODO: Support V2 when it's required
# from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1


def _cleanup_process(proc):
    """Terminate and join a child process if still alive."""
    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)


class PatchedGPUModelRunner(GPUModelRunnerV1):
    def load_model(self, load_dummy_weights: bool = False) -> None:
        print("Running load_model in PatchedGPUModelRunner")
        super().load_model(load_dummy_weights)
        print("Finished running load_model in PatchedGPUModelRunner")
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
        pp_rank = get_pp_group().rank_in_group
        pp_size = self.parallel_config.pipeline_parallel_size
        try:
            ep_rank = get_ep_group().rank_in_group
        except AssertionError as e:
            # EP group is only created for EP enabled models
            ep_rank = 0
        # https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#configuration
        # > Enable EP by setting the --enable-expert-parallel flag. The EP size is automatically calculated as:
        # > EP_SIZE = TP_SIZE × DP_SIZE
        ep_size = (
            self.parallel_config.tensor_parallel_size
            * self.parallel_config.data_parallel_size
        )
        gpu_id = torch.cuda.device(self.device).idx
        print(f"Running weight_updater_process with gpu_id {gpu_id}")
        self.mdl_distributed_proc = ctx.Process(
            target=weight_updater_process,
            args=(
                state_dict,
                num_heads,
                num_kv_heads,
                tp_rank,
                self.parallel_config.tensor_parallel_size,
                ep_rank,
                ep_size,
                gpu_id,
                pp_rank,
                pp_size,
            ),
            daemon=True,
        )
        # PP workers in vLLM are daemon processes, and Python's stdlib
        # forbids daemon processes from spawning children.  The check
        # lives in BaseProcess.start() and inspects
        #     _current_process._config.get('daemon')
        # Temporarily clear that flag so the weight-updater can start.
        _current = _mp.current_process()
        _was_daemon = _current._config.get("daemon", False)
        _current._config["daemon"] = False
        try:
            print("Starting weight_updater_process")
            self.mdl_distributed_proc.start()
            print("Finished starting weight_updater_process")
        finally:
            _current._config["daemon"] = _was_daemon
        atexit.register(_cleanup_process, self.mdl_distributed_proc)
