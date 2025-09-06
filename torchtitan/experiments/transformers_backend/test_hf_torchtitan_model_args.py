from transformers.models.llama.configuration_llama import LlamaConfig
from torchtitan.experiments.transformers_backend.model.hf_transformers_args import (
    HFTransformerModelArgs,
)
from torchtitan.config import JobConfig


def print_comparison_keys(ref_dict, tt_dict):
    all_keys = sorted(list(set(ref_dict.keys()) | set(tt_dict.keys())))
    print(f"{'Attribute':<30} | {'Original HF':<20} | {'TorchTitan HF':<20}")
    print("-" * 75)
    for key in all_keys:
        ref_val = ref_dict.get(key, "N/A")
        tt_val = tt_dict.get(key, "N/A")
        if str(ref_val) != str(tt_val):
            # Red for different
            print(f"\033[91m{key:<30} | {str(ref_val):<20} | {str(tt_val):<20}\033[0m")
        else:
            print(f"{key:<30} | {str(ref_val):<20} | {str(tt_val):<20}")

def compare_hf_tt_configs(model_name, flavor):
        ref_hf_config = LlamaConfig()
        
        model_args = HFTransformerModelArgs()
        job_config = JobConfig()
        job_config.model.name = model_name
        job_config.model.flavor = flavor
        model_args.update_from_config(job_config)
        tt_hf_config = model_args.convert_to_hf_config()

        ref_dict = ref_hf_config.to_dict()
        tt_dict = tt_hf_config.to_dict()

        try:
            assert ref_dict == tt_dict
            print(f"✅ Configs match for model name {model_name} with flavor: {flavor}")
        except AssertionError:
            print(f"❌ Configs do not match for model name {model_name} with flavor: {flavor}! Showing differences:")
            print_comparison_keys(ref_dict, tt_dict)
            raise

if __name__ == "__main__":
    model_names = [
        "meta-llama/Llama-3.2-1B",
    ]
    flavors = ["full"]

    for model_name in model_names:
        for flavor in flavors:
            print(f"\nTesting model name: {model_name} with flavor: {flavor}")
            compare_hf_tt_configs(model_name, flavor)