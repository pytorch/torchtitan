import argparse
import os

from torch.utils.data import DataLoader
from torchtrain.models import models_config, model_name_to_cls, model_name_to_tokenizer
from torchtrain.datasets import create_tokenizer, dataset_cls_map, pad_batch_to_longest_seq
from torch.distributed.device_mesh import init_device_mesh


def main(args):
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
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][args.model_conf]
    model_config.vocab_size = tokenizer.n_words

    model = model_cls.from_model_args(model_config)

    model.train()

    # for batch in data_loader:
    #     input_ids, labels = batch
    #     input_ids = input_ids.to(device_type)
    #     labels = labels.to(device_type)



    print(f">>> model: {model}, model_config: {model_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TorchTrain arg parser.')
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

    parser.add_argument('--model', type=str, default="llama", help="which model to train")
    parser.add_argument('--model_conf', type=str, default="debugmodel", help="which model config to train")
    parser.add_argument('--dataset', type=str, default="alpaca", help="dataset to use")
    parser.add_argument('--tokenizer_path', type=str, default="torchtrain/datasets/tokenizer.model", help="tokenizer path")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--tp_degree', type=int, default=LOCAL_WORLD_SIZE, help="Tensor/Sequence Parallelism degree")
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

    args = parser.parse_args()
    print(args)
    main(args)
