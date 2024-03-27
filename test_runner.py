# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import subprocess

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

CONFIG_DIR = "./train_configs"
for config_file in os.listdir(CONFIG_DIR):
    if config_file.endswith(".toml"):
        full_path = os.path.join(CONFIG_DIR, config_file)
        with open(full_path, "rb") as f:
            config = tomllib.load(f)
            is_integration_test = config["job"].get("use_for_integration_test", False)
            if is_integration_test:

                cmd = f"CONFIG_FILE={full_path} NGPU=4 ./run_llama_train.sh"
                print(f"=====Integration test: {cmd}=====")
                result = subprocess.run(
                    [cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=True,
                )
                print(result.stdout)
