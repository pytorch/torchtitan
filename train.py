import argparse
import os
from dataclasses import dataclass, field
from typing import List
import logging
from logging import getLogger
import sys  # for logging
# torch imports
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from contextlib import contextmanager
import contextlib


# torchtrain related
from torchtrain.models import models_config, model_name_to_cls, model_name_to_tokenizer
from torchtrain.datasets import create_tokenizer, dataset_cls_map, pad_batch_to_longest_seq

def lprint(msg=""):
  print(f"Debug ++> {sys._getframe().f_back.f_lineno}: {msg}")

logger = getLogger()


@dataclass
class TrainState:
    step: int = 0
    current_loss: float = -1
    losses: List[float] = field(default_factory=list)


def rank0_log(msg):
    if torch.distributed.get_rank() == 0:
        logger.info(msg)

def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def build_optimizer(model, args):
    # build optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"optimizer {args.optimizer} not added")

    return optimizer


def main(args):
    init_logger()

    # only support cuda for now
    device_type = "cuda"
    # distributed init
    world_size = int(os.environ["WORLD_SIZE"])
    dp_degree = world_size // args.tp_degree
    world_mesh = init_device_mesh(device_type, (dp_degree, args.tp_degree), mesh_dim_names=("dp", "tp"))

    model_name = args.model
    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, args.tokenizer_path)

    # build dataloader
    dataset_cls = dataset_cls_map[args.dataset]
    data_loader = DataLoader(dataset_cls(tokenizer), batch_size=args.batch_size, collate_fn=pad_batch_to_longest_seq)

    # build model
    # TODO: add meta initialization
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][args.model_conf]
    model_config.vocab_size = tokenizer.n_words

    model = model_cls.from_model_args(model_config)

    model.to(device_type)

    # build optimizer
    # TODO: add scheduler if needed
    optimizer = build_optimizer(model, args)

    # TODO: apply parallelisms, e.g. fsdp/tp
    # TODO: add profiler
    @contextlib.contextmanager
    def maybe_run_profiler(args, *pos_args, **kwargs):
        use_profiler: bool = args.run_profiler

        trace_dir = args.profile_folder
        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            rank0_log(f"exporting profile traces to {trace_dir}")
            prof.export_chrome_trace(
                f"{trace_dir}/rank{rank}_trace.json"
            )

        if use_profiler:
            if not os.path.exists(trace_dir):
                os.makedirs(trace_dir)

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=False,
                record_shapes=True,
            ) as torch_profiler:
                yield torch_profiler
        else:
            torch_profiler = contextlib.nullcontext()
            yield None

    if args.run_profiler:
        rank0_log(f"Profiling active.  Traces will be saved at {args.profile_folder}")


    # TODO: add metrics
    train_state = TrainState()

    # train loop
    model.train()

    with maybe_run_profiler(args) as torch_profiler:
        while train_state.step < args.steps or args.steps == -1:
            train_state.step += 1
            # get batch
            batch = next(iter(data_loader))
            input_ids, labels = batch
            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)

            # forward
            pred = model(input_ids)
            tok_loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1), reduction="none")
            loss = tok_loss.mean()

            # backward
            loss.backward()
            # TODO: add grad scaler

            # optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # if profiler is active
            if torch_profiler:
                torch_profiler.step()

            train_state.current_loss = loss.item()
            train_state.losses.append(train_state.current_loss)

            rank0_log(f"current loss: {train_state.current_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TorchTrain arg parser.')
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

    parser.add_argument('--model', type=str, default="llama", help="which model to train")
    parser.add_argument('--model_conf', type=str, default="debugmodel", help="which model config to train")
    parser.add_argument('--dataset', type=str, default="alpaca", help="dataset to use")
    parser.add_argument('--tokenizer_path', type=str, default="./torchtrain/datasets/tokenizer/tokenizer.model", help="tokenizer path")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--optimizer', type=str, default="AdamW", help="optimizer to use")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate to use")
    parser.add_argument('--steps', type=int, default=-1, help="how many train steps to run")
    parser.add_argument('--tp_degree', type=int, default=LOCAL_WORLD_SIZE, help="Tensor/Sequence Parallelism degree")
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--run_profiler', action='store_true', help='Whether to run the profiler.')
    parser.add_argument('--profile_folder', type=str, default="./torchtrain/profiler", help='Folder to save profile traces to.')
    args = parser.parse_args()
    main(args)
