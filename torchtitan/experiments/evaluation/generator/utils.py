# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
 
import os
import tyro

import torch

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import ConfigManager
from torchtitan.experiments.evaluation.generator.transformer import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator


PROJECT_ROOT = ""  # TODO: Change to the correct path if needed

SAVE_ROOT = ""  # TODO: Change to the correct path if needed

TOML = {
    "llama3_2_1b": "torchtitan/experiments/evaluation/llama3/train_configs/llama3.2_1b.toml",
    "llama3_2_3b": "torchtitan/experiments/evaluation/llama3/train_configs/llama3.2_3b.toml",
}


# Dummy classes for components not fully defined in the original script
class DummyOptimizerContainer: # Basic placeholder
    def init_cache_state_dict(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass


class DummyLRSchedulerContainer: # Basic placeholder
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass


def load_model_tokenizer(model_type_size, exp_name, max_seq_len: int = 8192, args: dict = None,):
    config_manager = ConfigManager()

    file_path = os.path.join(PROJECT_ROOT, TOML[model_type_size])
    cli_args = [f"--job.config_file={file_path}"]
    toml_values = config_manager._maybe_load_toml(cli_args)
    config_cls = config_manager._maybe_add_custom_args(cli_args, toml_values)

    base_config = (
        config_manager._dict_to_dataclass(config_cls, toml_values)
        if toml_values
        else config_cls()
    )
    custom_registry = tyro.constructors.ConstructorRegistry()
    job_config = tyro.cli(
        config_cls, args=cli_args, default=base_config, registry=custom_registry
    )

    # TODO: This c4_test is redundant.
    job_config.training.dataset = "c4_test"
    job_config.training.dataset_path = "/home/sangminbae/torchtitan/tests/assets/c4_test"

    job_config.training.max_seq_len = max_seq_len
    job_config.job.dump_folder = os.path.join(SAVE_ROOT, "outputs", exp_name)
    job_config.checkpoint.folder = os.path.join(SAVE_ROOT, "checkpoints", exp_name)
    job_config.checkpoint.load_step = -1
    job_config.checkpoint.enable_checkpoint = True

    # TODO: We only support single GPU evaluation for now
    dp_degree, dp_rank = 1, 0

    train_spec = train_spec_module.get_train_spec(job_config.model.name)

    # build model (using meta init)
    model_cls = train_spec.model_cls
    model_args = train_spec.model_args[job_config.model.flavor]

    tokenizer = (
        train_spec.build_tokenizer_fn(job_config)
        if train_spec.build_tokenizer_fn is not None
        else None
    )

    # set the model args from training job configs
    model_args.update_from_config(job_config)

    with torch.device("cuda"):
        model = model_cls(model_args)
    
    if args is not None and "dtype" in args:
        for param in model.parameters():
            if args["dtype"] == "bf16":
                param.data = param.data.to(torch.bfloat16)
            elif args["dtype"] == "fp32":
                param.data = param.data.to(torch.float32)
            else:
                raise ValueError(f"Unsupported dtype: {args['dtype']}")
        import gc; gc.collect(); torch.cuda.empty_cache()  # Clear cache after dtype conversion
    
    model = model.eval()

    dataloader = train_spec.build_dataloader_fn(
        dp_world_size=dp_degree,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        job_config=job_config,
    )
    dummy_optimizers = DummyOptimizerContainer()
    dummy_lr_schedulers = DummyLRSchedulerContainer()

    checkpointer = CheckpointManager(
        dataloader=dataloader,
        model_parts=[model],
        optimizers=dummy_optimizers,
        lr_schedulers=dummy_lr_schedulers,
        states={},
        checkpoint_config=job_config.checkpoint,
        sd_adapter=None,
        ft_manager=None,
    )

    try:
        checkpointer.load(step=job_config.checkpoint.load_step)
    except Exception as e:
        # If WORLD_SIZE is different, it might raise an error,
        # but checkpoint loading should still work
        print(f"DP degree mismatching error: {e}")

    return model, tokenizer


def load_transformer_generator(model_type_size, exp_name, max_seq_len: int = 8192, args: dict = None,):
    model, tokenizer = load_model_tokenizer(model_type_size, exp_name, max_seq_len, args)

    # get gen_args
    if args is None:
        args = {}

    gen_cfg = PackedCausalTransformerGeneratorArgs(**args,)
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    return generator