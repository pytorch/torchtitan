# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 4 generate.py

# use inference.sh "Your Question Here?" to run inference with a single prompt.

import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist

from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from transformers import AutoTokenizer

from torchtitan.tools.utils import Color

# Uncomment the model you want to run.
model_id, mesh_shape = "deepseek-ai/DeepSeek-V2-Lite-Chat", (1, 4)
# model_id, mesh_shape = "deepseek-ai/deepseek-v3", (8, 4)


def colorize_chat(text, user_color=None, assistant_color=None, output_color=None):
    """Parse and colorize chat output with optional colors for each role."""
    lines = text.split("\n")
    result = []

    current_role = None
    current_content = []

    def _process_current_content():
        if not current_role or not current_content:
            return None

        content = "\n".join(current_content)
        if current_role == "output":
            return (
                f"Output: {output_color}{content}{color.reset}"
                if output_color
                else f"Output: {content}"
            )
        else:
            try:
                prefix, rest = current_content[0].split(":", 1)
                role_color = user_color if current_role == "user" else assistant_color
                if role_color:
                    formatted = f"{prefix}:{role_color}{rest}{color.reset}"
                    if len(current_content) > 1:
                        formatted += (
                            f"{role_color}\n"
                            + "\n".join(current_content[1:])
                            + f"{color.reset}"
                        )
                    return formatted
            except ValueError:
                pass
        return content

    for line in lines:
        if line.startswith("Output:"):
            if processed := _process_current_content():
                result.append(processed)
            current_role = "output"
            content = line[len("Output:") :].strip()
            if output_color:
                content = f"Output: {output_color}{content}{color.reset}"
            else:
                content = f"Output: {content}"
            result.append(content)
            current_content = []

        elif line.startswith("User:"):
            if processed := _process_current_content():
                result.append(processed)
            current_role = "user"
            current_content = [line]

        elif line.startswith("Assistant:"):
            if processed := _process_current_content():
                result.append(processed)
            current_role = "assistant"
            current_content = [line]

        else:
            if current_content:
                current_content.append(line)
            elif line.strip() and current_role is None:
                # Handle system message at the beginning
                current_role = "output"
                if output_color:
                    result.append(f"Output: {output_color}{line.strip()}{color.reset}")
                else:
                    result.append(f"Output: {line.strip()}")

    # Process the last segment
    if processed := _process_current_content():
        result.append(processed)

    return "\n".join(result)


color = Color()


@dataclass
class DistConfig:
    mesh: DeviceMesh
    pp_mesh: DeviceMesh
    ep_mesh: DeviceMesh
    pp_size: int
    ep_size: int
    ep_rank: int
    pp_rank: int
    device: torch.device


def create_model(dist_config: DistConfig):
    model_args = deepseek_config_registry[model_id]
    model_args.ep_size = dist_config.ep_size
    model_args.num_stages = dist_config.pp_size
    model_args.stage_idx = dist_config.pp_rank
    model_args.max_seq_len = 4096  # 16384

    with dist_config.device, dist_config.mesh:
        model = DeepseekForCausalLM(model_args)
    load_weights_from_hf(model, model_id, dist_config.device)
    model.eval()
    model.setup_symm_mem(torch.bfloat16, dist_config.device)

    stage = PipelineStage(
        model,
        dist_config.pp_rank,
        dist_config.pp_size,
        dist_config.device,
        group=dist_config.pp_mesh.get_group(),
    )
    pp_schedule = ScheduleGPipe(stage, dist_config.pp_size)
    return model, pp_schedule


def create_dist_config(mesh: DeviceMesh):
    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    dist_config = DistConfig(
        mesh=mesh,
        pp_mesh=mesh["pp"],
        ep_mesh=mesh["ep"],
        pp_rank=mesh["pp"].get_local_rank(),
        pp_size=mesh["pp"].size(),
        ep_size=mesh["ep"].size(),
        ep_rank=mesh["ep"].get_local_rank(),
        device=device,
    )
    return dist_config


def decode(tokenizer, x):
    output = tokenizer.decode(x[0])
    # Clean up the output by removing special tokens
    bos = tokenizer.bos_token
    output = output.replace(bos, "")
    # Truncate at end of sentence token
    eos_token = tokenizer.eos_token
    if eos_token and eos_token in output:
        output = output.split(eos_token)[0]
    colored_output = colorize_chat(
        output,
        user_color=color.green,
        assistant_color=color.cyan,
        output_color=color.blue,
    )
    return colored_output


def time_generation(func):
    """Decorator to time generation functions and display timing and token per second results."""

    def wrapper(*args, **kwargs):
        rank = dist.get_rank()

        # Setup timer
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Call the original function
        result, tokens_generated = func(*args, **kwargs)

        # Record end time and calculate elapsed time
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)

        if rank == 0:
            print(
                f"\nGeneration time: {color.yellow}{elapsed_time / 1000:.2f} seconds{color.reset}"
            )
            print(f"Tokens generated: {color.blue}{tokens_generated}{color.reset}")
            print(
                f"Tokens per second: {color.green}{tokens_generated / (elapsed_time / 1000):.2f}\n{color.reset}"
            )

        return result

    return wrapper


@time_generation
@torch.inference_mode()
def generate(
    model,
    pp_schedule,
    tokenizer,
    dist_config,
    messages: list[dict],
    n_tokens: int = 200,
):
    rank = dist.get_rank()
    device = dist_config.device
    x = tokenizer.apply_chat_template(
        [messages] * dist_config.pp_size,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    next_idx = x.shape[-1]
    x = torch.cat([x, torch.zeros(x.shape[0], n_tokens, dtype=torch.int64)], dim=-1)
    x = x.to(device)

    tokens_generated = 0
    eos_token_id = tokenizer.eos_token_id
    # Create tensor on device for comparison
    eos_tensor = torch.tensor([eos_token_id], device=device)

    # Print initial progress indicator
    if rank == 0:
        print("Generating: ", end="", flush=True)

    for _ in range(n_tokens):
        if dist_config.pp_size > 1:
            if dist_config.pp_rank == 0:
                pp_schedule.step(x)
                torch.distributed.broadcast(
                    x,
                    group=dist_config.pp_mesh.get_group(),
                    group_src=dist_config.pp_size - 1,
                )
            elif dist_config.pp_rank == dist_config.pp_size - 1:
                preds = pp_schedule.step()
                next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
                x[:, next_idx] = next_token
                torch.distributed.broadcast(
                    x,
                    group=dist_config.pp_mesh.get_group(),
                    group_src=dist_config.pp_size - 1,
                )
                # Break if EOS token is generated (without .item())
                if torch.equal(next_token, eos_tensor):
                    tokens_generated += 1
                    break
            else:
                pp_schedule.step()
                torch.distributed.broadcast(
                    x,
                    group=dist_config.pp_mesh.get_group(),
                    group_src=dist_config.pp_size - 1,
                )

            next_idx += 1
        else:
            preds = model(x)
            next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
            x[:, next_idx] = next_token
            next_idx += 1
            # Break if EOS token is generated (without .item())
            if torch.equal(next_token, eos_tensor):
                tokens_generated += 1
                break

        tokens_generated += 1

        # Print progress indicator every 20 tokens
        if rank == 0 and tokens_generated % 20 == 0:
            print(f"{color.yellow}:{color.reset}", end="", flush=True)

    # Print newline after progress indicator
    if rank == 0:
        print()
        colored_output = decode(tokenizer, x)
        print(f"Without CUDA Graph:\n{colored_output}")

    return x, tokens_generated


@time_generation
@torch.inference_mode()
def generate_with_cuda_graph(
    model,
    tokenizer,
    dist_config,
    messages: list[dict],
    n_tokens: int = 100,
):
    rank = dist.get_rank()
    device = dist_config.device
    x = tokenizer.apply_chat_template(
        [messages] * dist_config.pp_size,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    next_idx = x.shape[-1]
    x = torch.cat([x, torch.zeros(x.shape[0], n_tokens, dtype=torch.int64)], dim=-1)
    x = x.to(device)

    torch.cuda.synchronize()
    eos_token_id = tokenizer.eos_token_id
    # Create tensor on device for comparison
    eos_tensor = torch.tensor([eos_token_id], device=device)

    # Create CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        preds = model(x)

    # Run CUDA graph
    tokens_generated = 0
    for _ in range(n_tokens):
        g.replay()
        next_token = torch.argmax(preds[:, next_idx - 1], dim=-1)
        x[:, next_idx] = next_token
        next_idx += 1
        tokens_generated += 1

        # Break if EOS token is generated (without .item())
        if torch.equal(next_token, eos_tensor):
            break

    if rank == 0:
        colored_output = decode(tokenizer, x)
        print(f"With CUDA Graph:\n{colored_output}")

    return x, tokens_generated


if __name__ == "__main__":
    # Get user prompt from command line arguments
    user_prompt = "What is 2+2?"  # Default prompt
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]

    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("pp", "ep"))
    rank = dist.get_rank()
    if rank == 0:
        print(
            f"{color.yellow}Running inference with {model_id} on {mesh_shape} mesh{color.reset}"
        )

    dist_config = create_dist_config(mesh)
    model, pp_schedule = create_model(dist_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    generate(model, pp_schedule, tokenizer, dist_config, messages)
    generate_with_cuda_graph(model, tokenizer, dist_config, messages)

    if rank == 0:
        print(f"\n{color.yellow}Closing inference mesh...{color.reset}")

    dist.destroy_process_group()
