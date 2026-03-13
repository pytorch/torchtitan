# flake8: noqa: F401
import vllm.v1.worker.gpu_model_runner
from torchtitan.grpo.vllm_handling.vllm_patching.lib_patching.gpu_model_runner import (
    PatchedGPUModelRunner,
)

# Patch GPU Runner at the source module so deferred imports in
# Worker.init_device() pick up the patched class.
vllm.v1.worker.gpu_model_runner.GPUModelRunner = PatchedGPUModelRunner

from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.gpu_worker import Worker

# Patch MP launcher with this context...
from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase

AsyncLLMEngine = AsyncLLM
