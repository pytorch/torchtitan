#!/bin/bash

# terminate on first error
set -e

# sweep over various important torchtitan + TE experiments

OUTPUT_FOLDER=/home/vasiliy/local/tmp/torchtitan_outputs
OUTPUT_LOGFILE=logs.txt

# need to loop over:
# 1. AC (none, full, selective with op)
# 2. experiment branches (TE and PT)

for AC_SETTING in none full
do

    for NAME in baseline te_ln_linear_f8 pt_f8 pt_f8_fsdp_f8
    do

        if [ $NAME == "baseline" ]; then
            EXTRA_ARGS="--training.compile --training.compile_linear"
        elif [ $NAME == "te_ln_linear_f8" ]; then
            EXTRA_ARGS="--training.te_swap_linear --training.te_float8_autocast"
        elif [ $NAME == "pt_f8" ]; then
            EXTRA_ARGS="--training.compile --training.compile_linear --float8.enable_float8_linear"
        elif [ $NAME == "pt_f8_fsdp_f8" ]; then
            EXTRA_ARGS="--training.compile --training.compile_linear --float8.enable_float8_linear --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp"
        else
            # should not get here
            exit 1
        fi

        # v6 contained an error
        OUTPUT_SUBFOLDER="20241204_v7_regional_linear_llama3_8b_name_${NAME}_ac_${AC_SETTING}"

        # create the subdir if does not exist, `tee` needs this
        mkdir -p $OUTPUT_FOLDER/$OUTPUT_SUBFOLDER

        CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh $EXTRA_ARGS \
            --job.dump_folder $OUTPUT_FOLDER/$OUTPUT_SUBFOLDER \
            --training.horizontally_fuse_fcs \
            --activation_checkpoint.mode $AC_SETTING \
            --activation_checkpoint.selective_ac_option 2 \
            --training.steps 200 \
            --profiling.profile_freq 100 2>&1 | tee $OUTPUT_FOLDER/$OUTPUT_SUBFOLDER/$OUTPUT_LOGFILE

    done

done
