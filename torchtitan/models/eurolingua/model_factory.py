

from functools import partial
import json
from pathlib import Path
import torch
import yaml
from torchtitan.models.eurolingua.args import GPT2LLMModelArgs
from torchtitan.models.eurolingua.gpt2_model import GPT2LLM
from dataclasses import asdict, dataclass
from typing import Annotated, Optional, Set

from pydantic import Field
from torchtitan.models.eurolingua.gpt2_model import AttentionConfig, AttentionImplementation, LayerNormWrapperConfig, PositionTypes
from torchtitan.models.eurolingua.model import ActivationType
from omegaconf import OmegaConf
from pydantic import BaseModel
import torch.nn as nn
import torch.distributed as dist
from datetime import datetime
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


class DebuggingArgs(BaseModel):
    logging_dir_path: Path
    tracked_ranks: list[int]


class GPT2LLMModelParsedArgs(BaseModel):
    sample_key: str
    prediction_key: str
    poe_type: PositionTypes
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[
        int, Field(strict=True, ge=1)
    ]  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    n_head_kv: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool  # True: bias in Linears like GPT-2. False: a bit better and faster
    attention_config: AttentionConfig
    attention_implementation: AttentionImplementation
    activation_type: ActivationType
    attention_norm_config: LayerNormWrapperConfig
    ffn_norm_config: LayerNormWrapperConfig
    lm_head_norm_config: LayerNormWrapperConfig
    use_weight_tying: bool
    debugging_args: Optional[DebuggingArgs] = None


def save_model_structure(model: nn.Module, model_structure_log_folder_path: str, tracked_ranks: Optional[list[int]] = None, tag: Optional[str] = None) -> None:
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if tracked_ranks is not None and rank in tracked_ranks:
        model_structure_log_folder_path = Path(model_structure_log_folder_path)
        model_structure_log = f"{'FQN':<60} {'Global Shape':<20} {'Local Shape':<20}\n"
        model_structure_log += "-" * 100 + "\n"

        for name, param in model.named_parameters():
            if isinstance(param, dist.tensor.DTensor):
                global_shape = tuple(param.shape)
                local_shape = tuple(param.to_local().shape)
            else:
                global_shape = tuple(param.shape)
                local_shape = ""            
            model_structure_log += f"{name:<60} {str(global_shape):<20} {str(local_shape):<20}\n"

        model_structure_log_folder_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = model_structure_log_folder_path / f"{timestamp}_{tag}_model_structure_rank_{rank}.txt"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(model_structure_log)

class ModelFactory:
    @staticmethod
    def get_gpt2_model(model_args: GPT2LLMModelArgs) -> nn.Module:
        config_path = model_args.model_config_path
        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        cfg = OmegaConf.create(config_dict)
        # Resolve interpolations
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

        model_args = GPT2LLMModelParsedArgs(**resolved_cfg)
        arg_dict = model_args.__dict__
        debugging_args = arg_dict.pop("debugging_args", None)
        model = GPT2LLM(**arg_dict)
        if debugging_args is not None:
            logging_dir_path = debugging_args.logging_dir_path
            tracked_ranks = set(debugging_args.tracked_ranks)
            model = ModelFactory._get_debugging_enriched_model(
                model=model,
                logging_dir_path=logging_dir_path,
                tracked_ranks=tracked_ranks,
            )


        return model


    @staticmethod
    def _get_debugging_enriched_model(
        model: nn.Module, logging_dir_path: Path, tracked_ranks: Optional[Set[int]] = None
    ) -> nn.Module:
        @dataclass
        class TensorStats:
            """Dataclass to hold tensor statistics."""
            global_shape: list[int]
            local_shape: list[int]
            is_dtensor: bool
            nan_count: int
            inf_count: int
            mean: float
            std: float
            min: float
            max: float


        @dataclass
        class CounterRef:
            value: int = 0

        rank = dist.get_rank() if dist.is_initialized() else 0

        if tracked_ranks is not None and rank not in tracked_ranks:
            return model
        if rank == 0:
            logging_dir_path.mkdir(parents=True, exist_ok=True)
        

        # timestamp yyyymmdd_hhmmss
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logging_file_path = logging_dir_path / f"{timestamp}_tensor_stats_rank_{rank}.jsonl"

        def get_tensor_stats(tensor: torch.Tensor) -> TensorStats:
            """Get statistics of a tensor."""
            local_tensor = tensor.to_local() if isinstance(tensor, dist.tensor.DTensor) else tensor
            float_dtypes = {torch.float, torch.bfloat16} 
            numeric_dtypes = float_dtypes | {torch.int, torch.long}

            dtype = local_tensor.dtype
            is_float = dtype in float_dtypes
            is_numeric = dtype in numeric_dtypes

            tensor_stats = TensorStats(
                global_shape=list(tensor.shape),
                local_shape=list(local_tensor.shape),
                is_dtensor=isinstance(tensor, dist.tensor.DTensor),
                nan_count=torch.isnan(local_tensor).sum().item(),
                inf_count=torch.isinf(local_tensor).sum().item(),
                mean=local_tensor.mean().item() if is_float else -1,
                std=local_tensor.std().item() if is_float else -1,
                min=local_tensor.min().item() if is_numeric else -1,
                max=local_tensor.max().item() if is_numeric else -1,
            )
            return tensor_stats

        def write_out_tensor_stats(tensor_stats: TensorStats, counter: int, hook_type: str, tensor_tag: str, rank: int):
            """Write out tensor statistics to a file."""
            with open(logging_file_path, "a", encoding="utf-8") as f:
                tensor_stats_dict = asdict(tensor_stats)
                tensor_stats_dict = {
                        "tensor_tag": tensor_tag,
                        "hook_type": hook_type,
                        **tensor_stats_dict,
                        "counter": counter,
                        "rank": rank,
                    }
            
                f.write(json.dumps(tensor_stats_dict) + "\n")

        def pre_forward_hook(module: nn.Module, forward_input, counter: CounterRef):
            if isinstance(forward_input, tuple):
                forward_inputs = forward_input
            else:
                forward_inputs = (forward_input,)

            for forward_input in forward_inputs:
                if isinstance(forward_input, dict):
                    for key, tensor in forward_input.items():
                        tensor_stats = get_tensor_stats(tensor)
                        write_out_tensor_stats(tensor_stats, counter.value, "forward_input", f"{module._debug_name}.{key}", rank)
                else:
                    tensor_stats = get_tensor_stats(forward_input)
                    write_out_tensor_stats(tensor_stats, counter.value, "forward_input", module._debug_name, rank)
            # Retrieves statistics of the module's parameters before forward pass.
            for name, param in module.named_parameters(recurse=False):
                tensor_stats = get_tensor_stats(param)
                full_name = f"{module._debug_name}.{name}"
                write_out_tensor_stats(
                    tensor_stats=tensor_stats,
                    counter=counter.value,
                    hook_type="forward_weights",
                    tensor_tag=full_name,
                    rank=rank,
                )
            counter.value += 1

        def forward_hook(module: nn.Module, forward_input, forward_output, counter: CounterRef):
            if isinstance(forward_output, tuple):
                forward_outputs = forward_output
            else:
                forward_outputs = (forward_output,)

            for out in forward_outputs:
                tensor_stats = get_tensor_stats(out)
                write_out_tensor_stats(tensor_stats, counter.value, "forward_output", module._debug_name, rank)
            counter.value += 1

        def backward_hook(module, grad_input, grad_output, counter: CounterRef):
            for grad_out in grad_output:
                tensor_stats = get_tensor_stats(grad_out)
                write_out_tensor_stats(tensor_stats, counter.value, "backward_output", module._debug_name, rank)
            counter.value += 1

        def register_hooks_recursively(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                child._debug_name = full_name
                # if rank 0 print
                if rank == 0:
                    print(f"Registering hooks for {full_name}")
                
                child.register_forward_pre_hook(partial(pre_forward_hook, counter=CounterRef()))
                child.register_forward_hook(partial(forward_hook, counter=CounterRef()))
                child.register_full_backward_hook(partial(backward_hook, counter=CounterRef()))
                register_hooks_recursively(child, full_name)

        register_hooks_recursively(model)

        return model
    

    @staticmethod
    def get_gpt2_tensor_parallelized_model(model: nn.Module, tp_mesh: DeviceMesh) -> nn.Module:
        model_tp_plan = {
            # Row-wise parallelism might seem counterintuitive here,
            # but the embedding layer has weight shape (vocab_size, n_embd).
            # Row-wise sharding allows each rank to store a slice of the vocabulary
            # and perform lookups only for the tokens it owns.
            # The input token IDs are replicated across all ranks so that each rank
            # can identify which tokens it is responsible for.
            # Each rank produces a partial embedding output, and an all-reduce is performed
            # in the background to obtain the full embedding vectors of shape
            # (batch_size, sequence_length, n_embd).
            # Finally, we shard the output on the sequence dimension to enable sequence parallelism
            # in the downstream transformer blocks.
            "transformer.wte": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "transformer.lm_head_norm": SequenceParallel(),
            "transformer.lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),  # Shard(-1) if loss parallelism is used
                use_local_output=True,  # default, should be not loss_parallel if loss parallelism is used
            ),
        }

        if isinstance(model.transformer.wpe, nn.Embedding):
            # If the position embedding is an nn.Embedding, we can shard it on the sequence dimension
            # to enable sequence parallelism in the downstream transformer blocks.
            # Note, for RoPE the wpe layer is an identity operation, which cannnot be sharded.
            model_tp_plan["transformer.wpe"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(0),
            )

        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan=model_tp_plan,
        )

        transformer_block_tp_plan = {
            "attention_norm": SequenceParallel(),
            "ffn_norm": SequenceParallel(),
            "attn": PrepareModuleInput(
                # here we prepare the actual input of the attention module
                # (i.e., the arguements to the forward method)
                # The incomming inputs are sharded on the sequence dimension
                # due to the pre-layer attention norm running sequence parallelism.
                # The inputs are transformed into the desired format by replicating
                # them across all ranks.
                # In the pytorch tutorial and torch titan we pass in an additional None argument
                # for freqs_cis (i.e., precomputed cosine and sine frequencies.), which is not
                # needed here due to implementation differences.
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.q_attn": ColwiseParallel(),
            "attn.k_attn": ColwiseParallel(),
            "attn.v_attn": ColwiseParallel(),
            "attn.c_proj": RowwiseParallel(output_layouts=Shard(1)),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.W": ColwiseParallel(),
            "mlp.W_2": RowwiseParallel(output_layouts=Shard(1)),
            "mlp.V": ColwiseParallel(),
        }

        for transformer_block in model.transformer.h:
            # override the number of q and kv heads
            if transformer_block.attn.n_head_q % tp_mesh.size() != 0:
                raise ValueError(
                    f"Number of query heads {transformer_block.attn.n_head_q} must be divisible by "
                    f"the number of tensor parallel devices {tp_mesh.size()}."
                )
            if transformer_block.attn.n_head_kv % tp_mesh.size() != 0:
                raise ValueError(
                    f"Number of key-value heads {transformer_block.attn.n_head_kv} must be divisible by "
                    f"the number of tensor parallel devices {tp_mesh.size()}."
                )
            transformer_block.attn.n_head_q = transformer_block.attn.n_head_q // tp_mesh.size()
            transformer_block.attn.n_head_kv = transformer_block.attn.n_head_kv // tp_mesh.size()
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=transformer_block_tp_plan,
            )

        return model