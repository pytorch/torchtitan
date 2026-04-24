import os
import sys
import time
from collections.abc import Mapping, Sequence

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
import importlib
from tqdm import tqdm

# --- Add the path to your dataloader module ---
# This ensures Python can find 'valeo_dataset.py'
VALEO_DATALOADER_PATH = "/p/project1/nxtaim-1/wunderlich3/valeo_dataloader/"
if VALEO_DATALOADER_PATH not in sys.path:
    sys.path.append(VALEO_DATALOADER_PATH)
    print(f"Added '{VALEO_DATALOADER_PATH}' to Python path.")


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not isinstance(config, Mapping):
        raise KeyError("Expected a mapping object for instantiation.")
    if "target" not in config:
        raise KeyError(f"Expected key `target` to instantiate in {config}")
    
    # Recursively instantiate nested configs
    def recursive_instantiate(x):
        if isinstance(x, Mapping):
            if "target" in x:
                return instantiate_from_config(x)
            return {k: recursive_instantiate(v) for k, v in x.items()}
        elif isinstance(x, Sequence) and not isinstance(x, str):
            return [recursive_instantiate(v) for v in x]
        return x

    target_cls = get_obj_from_str(config["target"])
    params = config.get("params", {})
    
    # Instantiate parameters before passing them to the target class
    instantiated_params = recursive_instantiate(params)
    
    return target_cls(**instantiated_params)


if __name__ == "__main__":
    # Path to your new test configuration file
    yaml_path = "config.yaml"
    print(f"Loading configuration from: {yaml_path}\n")
    
    cfg = OmegaConf.load(yaml_path)

    # Instantiate the DataLoader directly from the config
    print("Instantiating DataLoader...")
    start_time = time.perf_counter()
    dataloader = instantiate_from_config(cfg.test_dataloader)
    init_duration = time.perf_counter() - start_time
    print(f"DataLoader instantiated in {init_duration:.2f} seconds.")
    print(f"Dataset contains {len(dataloader.dataset)} items.")
    print("-" * 50)

    # --- Iterate and inspect a few batches ---
    num_batches_to_test = 3
    print(f"Fetching and inspecting {num_batches_to_test} batches...\n")

    for i, batch in enumerate(tqdm(dataloader, total=num_batches_to_test, desc="Testing batches")):
        print(f"\n--- Batch {i+1} ---")
        
        for key, value in batch.items():
            print(f"  Key: '{key}'")
            if isinstance(value, torch.Tensor):
                print(f"    Type: torch.Tensor")
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                print(f"    Min value: {value.min():.4f}")
                print(f"    Max value: {value.max():.4f}")
            elif isinstance(value, list):
                print(f"    Type: list")
                print(f"    Length: {len(value)}")
                if value:
                    print(f"    Element type: {type(value[0])}")
                    print(f"    First element: \"{str(value[0])[:80]}...\"") # Print a snippet
            else:
                print(f"    Type: {type(value)}")
        
        if i + 1 >= num_batches_to_test:
            break
            
    print("\n" + "-" * 50)
    print("Dataloader test completed successfully!")

