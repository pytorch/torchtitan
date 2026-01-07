# DeepSeek V3 Development Notes

## local-dev

### LoRA Fine-Tuning Workflow

Complete workflow for LoRA fine-tuning DeepSeek V3.

---

### 1. Download Model Weights

```bash
python scripts/download_hf_assets.py \
    --repo_id deepseek-ai/DeepSeek-V3.1-Base \
    --local_dir $BT_TEAM_CACHE_DIR \
    --all
```

Downloads to `$BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base/`.

---

### 2. Choose Your Training Config

#### Option A: Debug Model (2 layers, fast iteration)

Use `train_configs/deepseek_aghilora.toml` with flavor `deepseek_aghilora`:
- 2 transformer layers (`n_layers=2`)
- 1 dense layer (`n_dense_layers=1`)
- LoRA rank 16

```bash
NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_aghilora.toml" ./run_train.sh
```

#### Option B: Full 671B Model (production)

Use `train_configs/deepseek_v3_671b.toml` with flavor `671B`:
- 61 transformer layers
- 3 dense layers
- Add LoRA settings to the flavor in `__init__.py`

To enable LoRA on the 671B model, update the `671B` flavor in `__init__.py`:

```python
"671B": DeepSeekV3ModelArgs(
    # ... existing args ...
    finetune_lora_rank=16,     # Add this
    finetune_lora_alpha=32.0,  # Add this
),
```

Then run:

```bash
NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml" ./run_train.sh
```

**Tip**: Update `[checkpoint] folder` in the TOML to customize output location.

---

### 3. Create Matching HF Assets for Conversion

Before converting, you need an HF assets directory with a `config.json` matching your model architecture.

#### For Debug Model (deepseek_aghilora)

```bash
mkdir -p $BT_TEAM_CACHE_DIR/DeepSeek-v3.1-Base-DEBUG
cp $BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base/tokenizer*.json $BT_TEAM_CACHE_DIR/DeepSeek-v3.1-Base-DEBUG/
cp $BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base/*.py $BT_TEAM_CACHE_DIR/DeepSeek-v3.1-Base-DEBUG/
cp $BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base/generation_config.json $BT_TEAM_CACHE_DIR/DeepSeek-v3.1-Base-DEBUG/
```

Create `config.json` with these key differences from the original:
- `"num_hidden_layers": 2` (matches `n_layers=2`)
- `"first_k_dense_replace": 1` (matches `n_dense_layers=1`)

<details>
<summary>Full config.json for debug model</summary>

```json
{
  "architectures": ["DeepseekV3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV3Config",
    "AutoModel": "modeling_deepseek.DeepseekV3Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
  },
  "bos_token_id": 0,
  "eos_token_id": 1,
  "ep_size": 1,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "initializer_range": 0.02,
  "intermediate_size": 18432,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v3",
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 128,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 2,
  "num_key_value_heads": 128,
  "num_nextn_predict_layers": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0"
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 4,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.2",
  "use_cache": true,
  "v_head_dim": 128,
  "vocab_size": 129280
}
```

</details>

#### For Full 671B Model

Use the original HF assets directly—no modifications needed:

```bash
--hf_assets_path $BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base
```

---

### 4. Convert Checkpoint to HuggingFace Format

The conversion script merges LoRA weights into base weights automatically.

#### Debug model:

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
    outputs/checkpoint/step-50 \
    $BT_TEAM_CACHE_DIR/converted_hf_checkpoint \
    --model_name deepseek_v3 \
    --model_flavor deepseek_aghilora \
    --hf_assets_path $BT_TEAM_CACHE_DIR/DeepSeek-v3.1-Base-DEBUG \
    --export_dtype float16
```

#### Full 671B model:

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
    outputs/checkpoint/step-500 \
    $BT_TEAM_CACHE_DIR/converted_hf_checkpoint \
    --model_name deepseek_v3 \
    --model_flavor 671B \
    --hf_assets_path $BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base \
    --export_dtype float16
```

---

### 5. Serve with vLLM

```bash
vllm serve $BT_TEAM_CACHE_DIR/converted_hf_checkpoint --tensor-parallel-size 8
```

Test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "converted_hf_checkpoint",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---
---

## production

### Full 671B LoRA Fine-Tuning & Deployment

Production workflow for fine-tuning the complete DeepSeek V3 671B model.

---

### 1. Download Model Weights

```bash
python scripts/download_hf_assets.py \
    --repo_id deepseek-ai/DeepSeek-V3.1-Base \
    --local_dir $BT_TEAM_CACHE_DIR \
    --all
```

---

### 2. Use the 671B_lora Flavor

A pre-configured `671B_lora` flavor exists in `__init__.py` (identical to `671B` but with LoRA enabled):
- `finetune_lora_rank=16`
- `finetune_lora_alpha=32.0`

---

### 3. Configure Training

Use `train_configs/deepseek_v3_671b_lora.toml` (or edit as needed):

```toml
[model]
flavor = "671B_lora"
hf_assets_path = "/root/.cache/team_artifacts/DeepSeek-V3.1-Base"

[checkpoint]
folder = "outputs/671b_lora_checkpoint"
interval = 500
```

---

### 4. Run Training

#### Single Node (8 GPUs)

```bash
NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b_lora.toml" ./run_train.sh
```

#### Multi-Node (e.g., 4 nodes × 8 H200s = 32 GPUs)

Run on **each node**, setting `NODE_RANK` appropriately (0, 1, 2, 3):

```bash
MASTER_ADDR="<master-node-ip>"  # IP of node 0
MASTER_PORT=29500
NODE_RANK=0  # Set to 0, 1, 2, 3 on each respective node

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=${NODE_RANK} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m torchtitan.train \
    --job.config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b_lora.toml
```

---

### 5. Convert to HuggingFace Format

Use the **original** HF assets path (no config.json modifications needed for full model):

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
    outputs/671b_lora_checkpoint/step-500 \
    $BT_TEAM_CACHE_DIR/deepseek-v3-finetuned \
    --model_name deepseek_v3 \
    --model_flavor 671B_lora \
    --hf_assets_path $BT_TEAM_CACHE_DIR/DeepSeek-V3.1-Base \
    --export_dtype float16
```

This merges LoRA adapters into base weights and exports a standalone HF checkpoint.

---

### 6. Deploy with vLLM

```bash
vllm serve $BT_TEAM_CACHE_DIR/deepseek-v3-finetuned --tensor-parallel-size 8
```

Test endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3-finetuned",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```
