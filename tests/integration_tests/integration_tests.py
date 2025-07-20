# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class TestCaseConfigs:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    supported_models: List[str] = field(default_factory=lambda: ["llama3"])  # Default to llama3

    def __repr__(self):
        return self.test_descr



def build_core_functionality_tests() -> List[TestCaseConfigs]:
    """
    Build a dictionary of core functionality test configurations.
    This test suite is aimed at testing the core functionality and components of torchtitan.
    Core functionality tests only run on llama3 model.
    
    Returns:
        A list where each item is a TestCaseConfigs object
    """
    core_tests = []
    
    core_tests.extend([
        TestCaseConfigs(
            [
                [
                    "--profiling.enable_profiling",
                    "--metrics.enable_tensorboard",
                ],
            ],
            "default",
            "default",
        ),
        TestCaseConfigs(
            [
                [
                    "--compile.enable",
                ],
            ],
            "1D compile",
            "1d_compile",
        ),
        TestCaseConfigs(
            [
                [
                    "--compile.enable",
                    "--activation_checkpoint.mode selective",
                    "--activation_checkpoint.selective_ac_option op",
                ],
            ],
            "1D compile with selective op AC",
            "1d_compile_sac_op",
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable_checkpoint",
                ],
                [
                    "--checkpoint.enable",
                    "--training.steps 20",
                ],
            ],
            "Checkpoint Integration Test - Save Load Full Checkpoint",
            "full_checkpoint",
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.folder hf_checkpoint",
                    "--checkpoint.last_save_model_only",
                    "--checkpoint.last_save_in_hf",
                ],
                [
                    "--checkpoint.enable",
                    "--checkpoint.initial_load_path artifacts-to-be-uploaded/model_only_hf_checkpoint/hf_checkpoint/step-10/",
                    "--checkpoint.initial_load_model_only",
                    "--checkpoint.initial_load_in_hf",
                ],
            ],
            "Checkpoint Integration Test - save load model only checkpoint in HF definition and format",
            "model_only_hf_checkpoint",
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.last_save_model_only",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only fp32",
            "last_save_model_only_fp32",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--checkpoint.last_save_model_only",
                    "--checkpoint.export_dtype bfloat16",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only bf16",
            "last_save_model_only_bf16",
        ),
<<<<<<< HEAD:tests/integration_tests.py
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule InterleavedZeroBubble",
                ],
            ],
            "PP looped zero bubble test",
            "pp_looped_zero_bubble",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
                ],
            ],
            "PP zero bubble test (v shaped)",
            "pp_zbv",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test 1F1B",
            "pp_1f1b",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test GPipe",
            "pp_gpipe",
            ngpu=2,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 4",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP 1F1B 2D test",
            "pp_dp_1f1b",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP GPipe 2D test",
            "pp_dp_gpipe",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+TP 2D test",
            "pp_tp",
        ),
        OverrideDefinitions(
            [
                [
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--training.steps 20",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+DP+TP 3D test with save/load resume ckpt",
            "pp_dp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "PP+DP+TP 3D test with torch.compile",
            "3d_compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 1",
                ],
            ],
            "PP looped 1F1B test",
            "pp_looped_1f1b",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule PipelineScheduleMulti",
                    "--parallelism.pipeline_parallel_schedule_csv ./tests/assets/custom_schedule.csv",
                ],
            ],
            "PP with custom pipeline schedule loaded from CSV file",
            "pp_custom_csv",
            ngpu=2,
        ),
        OverrideDefinitions(
=======
        TestCaseConfigs(
>>>>>>> e3b51d8 (add integration test):tests/integration_tests/integration_tests.py
            [
                [
                    "--optimizer.name AdamW --optimizer.implementation foreach",
                ]
            ],
            "Foreach Optimizer Test",
            "optimizer_foreach",
            ngpu=2,
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable_checkpoint",
                ],
                [
                    # placeholder for the generation script's generate step
                ],
            ],
            "Generation script test",
            "test_generate",
            ngpu=2,
        ),
        TestCaseConfigs(
            [
                [
                    "--model.converters float8",
                    "--float8.enable_fsdp_float8_all_gather",
                    "--float8.precompute_float8_dynamic_scale_for_fsdp",
                    "--float8.emulate",
                ],
            ],
            "Float8 emulation test",
            "float8_emulation",
        ),
        TestCaseConfigs(
            [
                [
                    # Local batch size = 8, and `ngpu=2`, so default
                    # global batch size = 8 * 2 = 16.
                    # To achieve 2 gradient accumulation steps, multiply
                    # default global batch size by 2. 16 * 2 = 32.
                    "--training.local_batch_size 8",
                    "--training.global_batch_size 32",
                ],
            ],
            "Gradient accumulation",
            "gradient_accumulation",
            ngpu=2,
        ),
        TestCaseConfigs(
            [
                [
                    "--memory_estimation.enabled",
                ]
            ],
            "FSDP2 Memory Tracking and Estimation",
            "fsdp2_memory_estimation",
            ngpu=2,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.fsdp_reshard_after_forward always",
                ],
            ],
            "Test always resharding after forward pass",
            "fsdp_reshard_always",
            ngpu=2,
        ),
    ])

    return core_tests


def build_model_parallelism_tests() -> Dict[str, List[TestCaseConfigs]]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will broadly cover
    all the supported model parallelism patterns on all the supported models.
    
    Returns:
        A dictionary where:
        - key is the model name
        - value is a list of TestCaseConfigs
    """
    parallelism_tests = defaultdict(list)
    
    # ===== LLAMA3 MODEL PARALLELISM TESTS =====
    parallelism_tests["debug_model.toml"].extend([
        TestCaseConfigs(
            [
                [
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D eager",
            "2d_eager",
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=4",
                ]
            ],
            "DDP",
            "ddp",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "TP Only",
            "tp_only",
        ),
        TestCaseConfigs(
            [
                [
                    "--training.compile",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "2D compile",
            "2d_compile",
        ),
        # TestCaseConfigs(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 4",
        #             "--parallelism.pipeline_parallel_schedule InterleavedZeroBubble",
        #         ],
        #     ],
        #     "PP looped zero bubble test",
        #     "pp_looped_zero_bubble",
        #     ngpu=4,
        # ),
        # TestCaseConfigs(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
        #         ],
        #     ],
        #     "PP zero bubble test (v shaped)",
        #     "pp_zbv",
        #     ngpu=2,
        # ),
        # TestCaseConfigs(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_schedule 1F1B",
        #             "--parallelism.data_parallel_shard_degree 1",
        #         ],
        #     ],
        #     "PP 1D test 1F1B",
        #     "pp_1f1b",
        #     ngpu=2,
        # ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 1",
                ],
            ],
            "PP 1D test GPipe",
            "pp_gpipe",
            ngpu=2,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 4",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP 1F1B 2D test",
            "pp_dp_1f1b",
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule GPipe",
                    "--parallelism.data_parallel_shard_degree 2",
                ],
            ],
            "PP+DP GPipe 2D test",
            "pp_dp_gpipe",
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+TP 2D test",
            "pp_tp",
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--training.steps 20",
                    "--checkpoint.enable_checkpoint",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "PP+DP+TP 3D test with save/load resume ckpt",
            "pp_dp_tp",
            ngpu=8,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.compile",
                ],
            ],
            "PP+DP+TP 3D test with torch.compile",
            "3d_compile",
            ngpu=8,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                ],
                [
                    "--parallelism.pipeline_parallel_degree 4",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.pipeline_parallel_layers_per_stage 1",
                ],
            ],
            "PP looped 1F1B test",
            "pp_looped_1f1b",
            ngpu=4,
        ),
        # TestCaseConfigs(
        #     [
        #         [
        #             "--parallelism.pipeline_parallel_degree 2",
        #             "--parallelism.pipeline_parallel_schedule PipelineScheduleMulti",
        #             "--parallelism.pipeline_parallel_schedule_csv ./tests/assets/custom_schedule.csv",
        #         ],
        #     ],
        #     "PP with custom pipeline schedule loaded from CSV file",
        #     "pp_custom_csv",
        #     ngpu=2,
        # ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=4",
                ]
            ],
            "DDP",
            "ddp",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=4",
                    "--activation_checkpoint.mode='full'",
                    "--model.flavor=debugmodel_flex_attn",
                ]
            ],
            "FSDP+FLEX_ATTN",
            "fsdp+flex_attn",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.context_parallel_degree=4",
                    "--parallelism.context_parallel_rotate_method='allgather'",
                ]
            ],
            "CP (allgather)",
            "cp_allgather",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.context_parallel_degree=4",
                    "--parallelism.context_parallel_rotate_method='alltoall'",
                ]
            ],
            "CP (alltoall)",
            "cp_alltoall",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                ]
            ],
            "HSDP+TP",
            "hsdp+tp",
            ngpu=8,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "FSDP+CP",
            "fsdp+cp",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=1",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP (without dp_shard)",
            "hsdp+cp_without_dp_shard",
            ngpu=4,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "HSDP+CP (with dp_shard)",
            "hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        TestCaseConfigs(
            [
                [
                    "--parallelism.data_parallel_shard_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ]
            ],
            "FSDP+TP+CP",
            "fsdp+tp+cp",
            ngpu=8,
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--training.enable_cpu_offload",
                    "--optimizer.early_step_in_backward",
                ],
                [
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--training.enable_cpu_offload",
                    "--optimizer.early_step_in_backward",
                ],
            ],
            "Enable CPU Offload, Optimizer in backward with TP, DP, CP",
            "cpu_offload+opt_in_bwd+TP+DP+CP",
            ngpu=8,
            
        ),
        TestCaseConfigs(
            [
                [
                    "--checkpoint.enable_checkpoint",
                    "--training.steps 10",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--checkpoint.enable_checkpoint",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                ],
            ],
            "Optional checkpoint",
            "optional_checkpoint",
        ),
        TestCaseConfigs(
            [
                [
                    "--validation.enabled",
                    "--validation.dataset c4_test",
                    "--parallelism.data_parallel_replicate_degree=2",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.context_parallel_degree=2",
                ],
            ],
            "Validation test with fsdp, tp, cp",
            "validation_fsdp_tp_cp",
            ngpu=8,
        ),
    ])

    return parallelism_tests


def _run_cmd(cmd):
    return subprocess.run([cmd], text=True, shell=True)


def run_single_test(test_flavor: TestCaseConfigs, full_path: str, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--job.dump_folder {output_dir}/{test_name}"
    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        cmd = f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} ./run_train.sh"
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd
        if test_name == "fsdp2_memory_estimation":
            cmd = (
                f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
                "./scripts/estimate/run_memory_estimation.sh"
            )
        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)
        logger.info(
            f"=====Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )

        # save checkpoint (idx == 0) and load it for generation (idx == 1)
        if test_name == "test_generate" and idx == 1:
            cmd = (
                f"CONFIG_FILE={full_path} NGPU={test_flavor.ngpu} LOG_RANK={all_ranks} "
                f"CHECKPOINT_DIR={output_dir}/{test_name}/checkpoint/step-10 "
                "PROMPT='What is the meaning of life?' "
                f"./scripts/generate/run_llama_generate.sh --out > {output_dir}/{test_name}/generated_output.json"
            )

        result = _run_cmd(cmd)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args):
    # Determine which test suites to run based on the command-line argument
    assert args.config_path.endswith(".toml"), "Base config path must end with .toml"
    assert os.path.exists(args.config_path), f"Base config path {args.config_path} does not exist"
    
    # If user specifies a specific test name, the test_suite argument is ignored
    if args.test_name != "all":
        args.test_suite = TestSuite.ALL.value
    
    if args.test_suite == TestSuite.CORE.value:
        test_suites = {"core": build_core_functionality_tests()}
    elif args.test_suite == TestSuite.PARALLELISM.value:
        test_suites = {"parallelism": build_model_parallelism_tests()}
    else:  # ALL
        test_suites = {
            "core": build_core_functionality_tests(),
            "parallelism": build_model_parallelism_tests()
        }
    
    for test_suite_name, test_configs in test_suites.items():
        logger.info(f"Running {test_suite_name} test suite...")    
        with open(args.config_path, "rb") as f:
            config = tomllib.load(f)
            model_name = config["model"].get("name", "llama3")

            for test_flavor in test_configs[config_file]:
                # Filter by test name if specified
                if args.test != "all" and test_flavor.test_name != args.test:
                    continue
                
                # All tests now run on all models, no need to filter by model
                    
                # Check if we have enough GPUs
                    
                # Check if we have enough GPUs
                if args.ngpu < test_flavor.ngpu:
                    logger.info(
                        f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
                        f" because --ngpu arg is {args.ngpu}"
                    )
                else:
                    run_single_test(test_flavor, full_path, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to store test outputs")
    parser.add_argument(
        "--config_path", default="./tests/integration_tests/base_config.toml",
        help="Base config path for integration tests. This is the config that will be used as a base for all tests."
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="Specific test name to run (e.g., 'tp_only', 'full_checkpoint'). Use 'all' to run all tests (default: all)"
    )
    parser.add_argument(
        "--test_suite",
        default=TestSuite.ALL.value,
        choices=[ts.value for ts in TestSuite],
        help="Test suite to run: 'core' for TorchTitan core functionality tests, "
             "'parallelism' for model parallelism tests, or 'all' for both (default: all)"
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Specify the model to run tests on (default: llama3)"
    )

    parser.add_argument("--ngpu", default=8, type=int, help="Maximum number of GPUs to use")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")
    run_tests(args)


if __name__ == "__main__":
    main()
