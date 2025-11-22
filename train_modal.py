# /// script
# dependencies = [
#   "modal",
# ]
# ///
import modal 


app = modal.App("torchtitan-rl-training")

image = (modal.Image.debian_slim()
    .apt_install("pciutils")
    .uv_sync(gpu="A100:1")
    # .uv_pip_install("torch", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu129") # TODO: Attempt to install torch from nightly build
    .add_local_dir(".", "/root", 
        ignore=["*.pyc", "*.pyo", "*.pyd", "*.pyw", "*.pyz", ".venv/*", "docs/*",
         "outputs/*", "tests/*", ".ci/*", "benchmarks/*"]))
         

hf_cache_storage = modal.Volume.from_name("HF-CACHE", create_if_missing=True)

@app.function(image=image, gpu="A100:2")
def check():
    import subprocess
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run("ls")

@app.function(image=image, 
    gpu="A100-80GB:2", 
    timeout=3600, 
    secrets=[modal.Secret.from_name("huggingface-secret")], 
    volumes={"/HF-CACHE": hf_cache_storage},
    env={"HF_HOME": "/HF-CACHE"})
def train():
    import subprocess
    import os
    import torch
    print("Downloading tokenizer...")
    subprocess.run(['python', 'scripts/download_hf_assets.py', '--repo_id', 'meta-llama/Llama-3.1-8B', '--assets', 'tokenizer'], check=True)
    print("Running training...")
    env = os.environ.copy()
    env.update({
        "CONFIG_FILE": "./torchtitan/models/llama3/train_configs/llama3_8b.toml",
        "NGPU": str(torch.cuda.device_count()),
    })
    subprocess.run(['./run_train.sh'], check=True, env=env)



