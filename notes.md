python scripts/checkpoint_conversion/convert_to_hf.py \
    ./outputs/checkpoint/step-10 \
    ./outputs/hf_checkpoint \
    --hf_assets_path ./assets/hf/DeepSeek-V3.1-Base \
    --model_name deepseek_v3 \
    --model_flavor 671B \
    --export_dtype bfloat16


python scripts/checkpoint_conversion/convert_to_hf.py \
    /mnt/ckpts/checkpoint/step-5 \
    /mnt/ckpts/checkpoint/step-5-hf \
    --hf_assets_path ./assets/hf/deepseek-moe-16b-base \
    --model_name deepseek_v3 \
    --model_flavor 16B \
    --export_dtype bfloat16


for dir in /mnt/ckpts/checkpoint/*/; do
    dir_name=$(basename "$dir")
    if [[ ! "$dir_name" =~ -hf$ ]]; then
        python scripts/checkpoint_conversion/convert_to_hf.py \
            /mnt/ckpts/checkpoint/"$dir_name" \
            /mnt/ckpts/checkpoint/"$dir_name"-hf \
            --hf_assets_path ./assets/hf/deepseek-moe-16b-base \
            --model_name deepseek_v3 \
            --model_flavor 16B \
            --export_dtype bfloat16
    fi
done

DEBUG=1 DEBUG_WAIT_RANKS="0" NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh