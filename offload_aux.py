import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Callable, TypeVar, Any
import json
import torch
import os
import numpy as np
from torch.nn import functional as F
import random
from io import BytesIO


T = TypeVar("T")

def quantized_knapsack(values, weights, bag_weight):
    @lru_cache(None)
    def core(idx, weight) -> Tuple[List[int], int]:
        if idx == 0:
            return ([0], 0) if weight < weights[0] else ([1], values[0])
        else:
            curr_w = weights[idx]
            curr_v = values[idx]

            # take
            take_idx, take_val = core(idx - 1, weight - curr_w)
            take_val += curr_v
            take_idx = list(take_idx) + [1]

            # no take
            no_take_idx, no_take_val = core(idx - 1, weight)
            no_take_idx = list(no_take_idx) + [0]

            # assert len(no_take_idx) == idx + 1, f"no_take = {no_take_idx}, idx = {idx}"
            # assert len(take_idx) == idx + 1, f"take = {take_idx}, idx = {idx}"

            if no_take_val >= take_val:
                return no_take_idx, no_take_val
            else:
                return take_idx, take_val

    return core(len(values) - 1, bag_weight)


# pyre-unsafe
def naive_fmin(
    cannonical_names: List[str],
    current_qps_explorations,
    current_qps_explorations_labels,
    current_memory_explorations,
    current_memory_explorations_labels,
    max_allowed_memory,
    automatically_contract_global_decision_to_local_decisions=True,
) -> Tuple[Dict[str, int], bool]:
    # get rid of  additional probes, and this could be 0,0,0...,0
    # say 4 choices, must contain
    # 1, 1, 1, 1
    # 0, 1, 1, 1
    # 1, 0, 1, 1
    # 1, 1, 0, 1
    # 1, 1, 1, 0
    current_qps_explorations = current_qps_explorations[: len(cannonical_names) + 1]
    current_qps_explorations_labels = current_qps_explorations_labels[
        : len(cannonical_names) + 1
    ]
    current_memory_explorations = current_memory_explorations[
        : len(cannonical_names) + 1
    ]
    current_memory_explorations_labels = current_memory_explorations_labels[
        : len(cannonical_names) + 1
    ]

    if len(current_qps_explorations[0]) != len(current_memory_explorations[0]):
        if not automatically_contract_global_decision_to_local_decisions:
            assert (
                0
            ), f"ensure qps exploration and mem exploration matches in varibale sizes"
        else:
            assert len(
                current_memory_explorations[0]
            ) * torch.distributed.get_world_size() == len(current_qps_explorations[0])
            # collapse
            current_qps_explorations = [
                x[: len(current_memory_explorations[0])]
                for x in current_qps_explorations
            ]

    if torch.distributed.get_rank() == 0:
        logging.info(f"CURRENT_QPS_EXP = {current_qps_explorations}")
        logging.info(f"CURRENT_QPS_LBL = {current_qps_explorations_labels}")
        logging.info(f"CURRENT_MEM_EXP = {current_memory_explorations}")
        logging.info(f"CURRENT_MEM_EXP = {current_memory_explorations_labels}")
        logging.info(f"cannonical_names = {cannonical_names}")

    for onehot_idx in range(len(cannonical_names)):
        one_hot = [1] * len(cannonical_names)
        one_hot[onehot_idx] = 0
        # logging.info(
        #     f"current_qps_explorations = {current_qps_explorations}, onehot = {one_hot}"
        # )
        if one_hot not in current_qps_explorations:
            best = {}
            for idx, name in enumerate(cannonical_names):
                best[name] = one_hot[idx]

            return best, False

    # everything has been tried. Now solve a knapsack
    base_trial = [1] * len(cannonical_names)
    base_qps_idx = current_qps_explorations.index(base_trial)
    base_mem_idx = current_memory_explorations.index(base_trial)
    assert base_qps_idx == base_mem_idx

    base_qps_val = current_qps_explorations_labels[base_qps_idx]
    base_mem_cost = current_memory_explorations_labels[base_mem_idx]

    weights = []
    values = []

    for onehot_idx in range(len(cannonical_names)):
        one_hot = [1] * len(cannonical_names)
        one_hot[onehot_idx] = 0

        qps_idx = current_qps_explorations.index(one_hot)
        mem_idx = current_memory_explorations.index(one_hot)
        assert qps_idx == mem_idx

        val = max(
            0, base_qps_val - current_qps_explorations_labels[qps_idx]
        )  # value: the amount of qps gain
        weight = max(
            0, int((current_memory_explorations_labels[mem_idx] - base_mem_cost) * 1000)
        )  # cost: the amount of weight

        weights.append(weight)
        values.append(val)

    if torch.distributed.get_rank() == 0:
        logging.info(f"KNAPSACK. WEIGHTS = {weights}")
        logging.info(f"KNAPSACK. VALUES = {values}")

    assert len(values) == len(
        weights
    ), f"{values} vs {weights}, max_allowed_memory = {max_allowed_memory}"

    ret_idx, ret_val = quantized_knapsack(
        values, weights, max_allowed_memory - base_mem_cost
    )
    # 1 in knapsack means offload is OFF
    ret_idx = [1 - x for x in ret_idx]

    assert len(ret_idx) == len(values), f"{ret_idx} vs {values}"

    best = {}

    if torch.distributed.get_rank() == 0:
        logging.info(
            f"KNAPSACK. SOLUTION = {ret_idx}, REWARD = {ret_val}, BAG = {max_allowed_memory}"
        )

    for idx, name in enumerate(cannonical_names):
        best[name] = ret_idx[idx]

    return best, True


def forward_offload_to_cpu(
    orig_fwd: Callable[..., T],
) -> Callable[..., T]:
    def offloaded_fwd(*args: Any, **kwargs: Any) -> T:
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return orig_fwd(*args, **kwargs)

    return offloaded_fwd


def digest_module_offload(module_sequencer, module_offloaded):
    ret = []
    for key in module_sequencer:
        ret.append(module_offloaded[key])

    return tuple(ret)


def greedy_descend(
    module_sequencer, module_offloaded, max_items=50, include_hamming_configs=True
):
    max_items = min(max_items, len(module_sequencer))
    ret = []
    for _ in range(max_items):
        new_cfg = {}
        for name in module_sequencer:
            dropout = random.random() < 0.5
            val = 0 if dropout else int(module_offloaded[name])
            new_cfg[name] = val

        if new_cfg != module_offloaded:
            ret.append(new_cfg)

    for key in module_sequencer:
        if module_offloaded[key] == 1:
            # create a copy
            cp = dict(module_offloaded)
            cp[key] = 0
            ret.append(cp)

    return [dict(t) for t in {tuple(d.items()) for d in ret}]


class PassThroughModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        return args if len(args) > 1 else args[0]


class SkippableBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, args):
        if args.shape[0] > 1:
            return super().forward(args)
        else:
            return args

class ThreeLayerMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dtype):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, device=device, dtype=dtype)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)  # Added hidden layer
        self.fc3 = torch.nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Apply the second hidden layer
        x = self.relu(x)
        x = self.fc3(x)
        x += residual
        return x

class PerformanceModelModule(torch.nn.Module):
    def __init__(
        self,
        num_modules,
        use_dual_embedding_table,
        normalization_type,
        device,
        dtype,
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = 8
        self.use_dual_embedding_table = use_dual_embedding_table
        self.embedding_offload = torch.nn.Linear(
            num_modules,
            self.embedding_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if self.use_dual_embedding_table:
            self.embedding_offload_off = torch.nn.Linear(
                num_modules,
                self.embedding_dim,
                bias=False,
                device=device,
                dtype=dtype,
            )

        self.num_modules = num_modules
        if normalization_type == "layer_norm":
            self.layernorm_emb = torch.nn.LayerNorm(
                [self.embedding_dim]
                if not self.use_dual_embedding_table
                else 2 * self.embedding_dim,
                device=device,
                dtype=dtype,
            )
            self.layernorm = (
                PassThroughModule()
            )  # torch.nn.LayerNorm([self.mlp_arch[-1]], device=device)
        elif normalization_type == "batch_norm":
            self.layernorm_emb = SkippableBatchNorm1d(
                self.embedding_dim
                if not self.use_dual_embedding_table
                else 2 * self.embedding_dim,
                device=device,
                dtype=dtype,
            )
            self.layernorm = (
                PassThroughModule()
            )  # SkippableBatchNorm1d(self.mlp_arch[-1], device=device)
        elif normalization_type == "tanh":
            self.layernorm_emb = torch.nn.Tanh()
            self.layernorm = PassThroughModule()
        elif normalization_type == "relu":
            self.layernorm_emb = torch.nn.Tanh()
            self.layernorm = PassThroughModule()
        elif normalization_type == "none" or normalization_type is None:
            self.layernorm_emb = PassThroughModule()
            self.layernorm = PassThroughModule()
        else:
            assert 0, f"unknown normalization type {normalization_type}"

        logging.info(
            f"normalization type = {normalization_type}, use_dual_embedding_table = {use_dual_embedding_table}"
        )

        # self.dense_module = ResidualMLP(
        #     mlp_arch=self.mlp_arch,
        #     activation=normalization_type
        #     if normalization_type is not None
        #     and normalization_type != "none"
        #     and normalization_type != "batch_norm"
        #     and False
        #     else "relu",
        # )
        input_dim = self.embedding_dim
        if self.use_dual_embedding_table:
            input_dim *= 2

        self.dense_module = ThreeLayerMLP(
            input_dim,
            input_dim * 2,
            input_dim,
            device,
            dtype,
        )
        # self.mlp_arch[-1]
        self.proj = torch.nn.Linear(input_dim, 1, device=device, dtype=dtype)

    def forward(self, input_embs):
        bs, dim = input_embs.shape
        assert (
            dim == self.num_modules
        ), f"input_dim = {dim} vs modules = {self.num_modules}"

        # logging.info(f"input_embs = {input_embs.shape}")
        embedding_on = self.embedding_offload(input_embs)
        if self.use_dual_embedding_table:
            embedding_off = self.embedding_offload_off(1 - input_embs)
            embeddings = torch.cat([embedding_on, embedding_off], dim=1)
        else:
            embeddings = embedding_on

        emb_ln = self.layernorm_emb(embeddings)

        # assert 0, f"emb_ln = {emb_ln.shape}"
        predicted = self.dense_module(emb_ln)
        layernormed = self.layernorm(predicted)
        return self.proj(layernormed)


class PerformanceModelModule1(torch.nn.Module):
    def __init__(
        self,
        num_modules,
        use_dual_embedding_table,
        normalization_type,
        noramlization_value,
        rank_loss_margin,
        device,
        dtype,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.params = torch.nn.Parameter(
            torch.rand([2 * num_modules + 1], device=device, dtype=dtype)
        )  # on plus off plus a bias term

    def forward(self, input_embs):
        oneoff = 1 - input_embs
        bias_on = torch.ones(
            [input_embs.shape[0], 1], device=input_embs.device, dtype=input_embs.dtype
        )
        ensembled = torch.cat([oneoff, input_embs, bias_on], dim=1)
        assert torch.all(torch.sum(ensembled, dim=1) == self.num_modules + 1), ensembled

        embeddings = ensembled * self.params
        ret = torch.sum(embeddings, dim=1).unsqueeze(1)
        return ret


class NoopScheduler:
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass


class PerformanceModelTrainer:
    def __init__(
        self,
        name,
        num_embeddings,
        normalization_value,
        rank_loss_margin,
        use_dual_embedding_table,
        normalization_type,
        collapse_inputs_by_world_size,
        device,
        dtype=torch.bfloat16,
    ):
        module = PerformanceModelModule(
            num_embeddings,
            use_dual_embedding_table,
            normalization_type,
            device,
            dtype,
        )
        self.num_embeddings = num_embeddings
        self.module = module
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=1e-2, amsgrad=True
        )
        self.detection_window = 50000
        self.max_updates = 5000
        # torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.max_updates
        )
        self.device = device
        self.dtype = dtype
        self.id = name
        self.normalization_type = normalization_type
        self.use_dual_embedding_table = use_dual_embedding_table
        self.input_sample = None
        self.normalization_value = normalization_value
        self.rank_loss_margin = rank_loss_margin
        self.collapse_inputs_by_world_size = collapse_inputs_by_world_size

    def train_once(
        self,
        activated_indices: list,
        groundtruth_one: float,
        rankloss_output1_larger_value: bool,
    ):
        # with autograd.detect_anomaly():

        input_embs = self._preproc(activated_indices)

        # input_embs: [bs, num_embeddings]
        actual = (
            torch.tensor(
                groundtruth_one,
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(1)
            * self.normalization_value
        )

        marginal_loss_margin = self.rank_loss_margin

        updates_rem = self.max_updates
        # create rank_loss
        while updates_rem > 0:
            updates_rem -= 1
            self.optimizer.zero_grad()
            output_one_og = self.module(input_embs)
            output_one = output_one_og

            assert (
                output_one.shape == actual.shape
            ), f"o1 = {output_one.shape}, actual = {actual.shape}, input = {activated_indices}, gt = {groundtruth_one}"

            loss = torch.abs(output_one - actual)  # * loss_scaler
            loss_one = torch.sum(loss)

            output_one_og = output_one_og.detach().clone()

            # create loss2, which produces a rankloss
            hi = 1
            rand_diff = torch.randint_like(input_embs, low=0, high=hi + 1)
            rand_input = torch.maximum(
                torch.zeros_like(input_embs), input_embs - rand_diff
            )
            assert torch.all(rand_input <= input_embs)

            output_two = self.module(rand_input)
            target = torch.ones(
                [output_two.shape[0], 1],
                device=actual.device,
                dtype=input_embs.dtype,
            )
            if rankloss_output1_larger_value is False:
                # no, rank2 has larger value
                target *= -1

            loss_two = F.margin_ranking_loss(
                output_one_og,
                output_two,
                target,
                reduction="mean",
                margin=marginal_loss_margin,
            )

            rand_diff_on = torch.randint_like(input_embs, low=0, high=hi + 1)
            multiplier = (
                1
                if not self.collapse_inputs_by_world_size
                else torch.distributed.get_world_size()
            )
            rand_input_on = torch.minimum(
                torch.ones_like(input_embs) * multiplier, input_embs + rand_diff_on
            )
            assert torch.all(rand_input_on >= input_embs)
            output_three = self.module(rand_input_on)
            # if target is negative, there is ranking loss if output_three < output_one
            loss_three = F.margin_ranking_loss(
                output_three,
                output_one_og,
                target,
                reduction="mean",
                margin=marginal_loss_margin,
            )
            # if UPDATE_TIMES % 10 == 0:
            #     print(f"loss_one = {loss_one}, loss_two = {loss_two}, loss_three = {loss_three}")
            loss = torch.sum(loss_one + loss_two + loss_three)
            # assert not torch.any(torch.isnan(loss))
            # assert not torch.any(torch.isnan(loss_one))
            # assert not torch.any(torch.isnan(loss_two))
            # assert not torch.any(torch.isnan(loss_three))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.detach(), self._postproc(output_one.detach())

    def _postproc(self, output):
        return output / (self.normalization_value + 1e-9)

    def _preproc(self, activated_indices: list):
        input_embs = torch.tensor(
            activated_indices, device=self.device, dtype=self.dtype
        )
        if self.collapse_inputs_by_world_size:
            world = torch.distributed.get_world_size()
            num_embs = input_embs.shape[1] // world
            assert (
                num_embs * world == input_embs.shape[1]
            ), f"{num_embs} x {world} should be {input_embs.shape[1]}"
            input_embs = input_embs.reshape(-1, world, num_embs)
            input_embs = torch.sum(input_embs, dim=1)

        return input_embs

    def inference_once(self, activated_indices: list):
        if self.input_sample is None:
            self.input_sample = (activated_indices,)

        # with autograd.detect_anomaly():
        with torch.no_grad():
            input_embs = self._preproc(activated_indices)
            output_one = self.module(input_embs)

        return self._postproc(
            output_one.item() if output_one.numel() == 1 else output_one
        )

    def create_bundle(self):
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_embeddings": self.num_embeddings,
            "use_dual_embedding_table": self.use_dual_embedding_table,
            "normalization_type": self.normalization_type,
            "input_sample": self.input_sample,
        }

    def load_from_bundle(self, checkpoint_bundle):
        # rematerialization
        self.inference_once(
            *checkpoint_bundle["input_sample"],
        )

        self.module.load_state_dict(checkpoint_bundle["model"])
        self.optimizer.load_state_dict(checkpoint_bundle["optimizer"])


class PerformanceModelTrainerXGB:
    def __init__(
        self,
        name,
        num_embeddings,
        normalization_value,
        rank_loss_margin,
        use_dual_embedding_table,
        normalization_type,
        collapse_inputs_by_world_size,
        device,
        dtype=torch.bfloat16,
    ):
        self.num_embeddings = num_embeddings
        self.headers = [f"module_{i}" for i in range(num_embeddings)]
        self.name = name
        self.current_model = None

    def train_once(
        self,
        activated_indices: list,
        groundtruth_one: float,
        rankloss_output1_larger_value: bool,
    ):
        self.current_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
        )

        activated_indices = np.array(activated_indices)
        groundtruth_one = np.array(groundtruth_one)
        self.current_model.fit(activated_indices, groundtruth_one)

        y_pred = self.current_model.predict(activated_indices)

        loss = np.abs(y_pred - groundtruth_one)

        return torch.sum(torch.from_numpy(loss)), torch.from_numpy(y_pred)

    def inference_once(self, activated_indices: list):
        activated_indices = np.array(activated_indices)
        np_results = torch.from_numpy(self.current_model.predict(activated_indices))
        return np_results.item() if np_results.numel() == 1 else np_results

    def create_bundle(self):
        return {}

    def load_from_bundle(self, checkpoint_bundle):
        pass


def digest_forward_name(fwd):
    fwd = str(fwd)
    return fwd if "(" not in fwd else fwd[: fwd.index("(")]


def create_file_with_path_for_write(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, 'w')

def dbg_upload_training_data_untransferrable_to_manifold(
    target_manifold_filename,
    target_manifold_foldername,
    inputs,
    labels,
    bucket_name="ai_ts_cp",
    target_manifold_root_dir="tree/copilot/training",
):
    dictionary = {
        "inputs": inputs,
        "labels": labels,
    }
    # logging.info(f"serializing object {dictionary}")
    # Serializing json
    # logging.info(
    #     f"inputs = {inputs}, labels = {labels}, types = {[[type(y) for y in x] for x in inputs]}, label types = {[type(x) for x in labels]}"
    # )
    target_manifold_dir = f"{target_manifold_root_dir}/{target_manifold_foldername}"
    ROOT = "/home/liangluo"
    target_object_name = ROOT + "/" +target_manifold_dir + "/" + target_manifold_filename
    target_object_name = target_object_name.replace("//", "/")

    # Writing to sample.json
    with create_file_with_path_for_write(target_object_name) as outfile:
        json.dump(dictionary, outfile)

def file_exists(file_path):
    return os.path.exists(file_path)


def dbg_download_training_data_untransferrable_from_manifold(
    target_manifold_filename,
    target_manifold_foldername,
    bucket_name="ai_ts_cp",
    target_manifold_root_dir="tree/copilot/training",
) -> dict:
    target_manifold_dir = f"{target_manifold_root_dir}/{target_manifold_foldername}"
    ROOT = "/home/liangluo"
    target_object_name = ROOT + "/" +target_manifold_dir + "/" + target_manifold_filename
    target_object_name = target_object_name.replace("//", "/")

    if file_exists(target_object_name):
        logging.info(f"Loading training data from {target_object_name}")
        with open(target_object_name, "r") as f:
            training_data = json.load(f)

        ip = training_data["inputs"]
        lbl = training_data["labels"]
    else:
        ip = []
        lbl = []

    return {
        "inputs": ip,
        "labels": lbl,
    }


def save_trainer(
    target_manifold_filename,
    target_manifold_rootname,
    model_trainer: PerformanceModelTrainer,
    bucket_name="ai_ts_cp",
    target_manifold_dir="tree/copilot",
):
    checkpoint_bundle = model_trainer.create_bundle()
    ROOT = "/home/liangluo"


    target_manifold_dir = f"{target_manifold_dir}/{target_manifold_rootname}"

    target_object_name = target_manifold_dir + "/" + target_manifold_filename
    target_object_name = (ROOT + "/" + target_object_name).replace("//", "/")
    with create_file_with_path_for_write(target_object_name):
        pass
    torch.save(checkpoint_bundle, target_object_name)

def get_trainer(
    # create a new model or load from manifold
    model_type,
    target_manifold_filename,
    target_manifold_rootname,
    name,
    num_embeddings,
    normalization_value,
    rank_loss_margin,
    use_dual_embedding_table,
    normalization_type,
    collapse_inputs_by_world_size,
    device,
    bucket_name="ai_ts_cp",
    target_manifold_dir="tree/copilot",
):
    target_manifold_dir = f"{target_manifold_dir}/{target_manifold_rootname}"
    ROOT = "/home/liangluo"

    target_object_name = ROOT + "/" + target_manifold_dir + "/" + target_manifold_filename
    # create a stub
    model_trainer = model_type(
        name=name,
        num_embeddings=num_embeddings,
        normalization_value=normalization_value,
        rank_loss_margin=rank_loss_margin,
        use_dual_embedding_table=use_dual_embedding_table,
        normalization_type=normalization_type,
        collapse_inputs_by_world_size=collapse_inputs_by_world_size,
        device=device,
    )

    if file_exists(target_object_name):
        logging.info(f"Loading model from {target_object_name}")
        checkpoint_bundle = torch.load(target_object_name)
        model_trainer.load_from_bundle(checkpoint_bundle)
    else:
        logging.info("no exisiting models found.")

    return model_trainer


def toggle_offload_simple(module, _legokit_original_fwd, on, already_on):
    if on:
        if not already_on:
            module.forward = forward_offload_to_cpu(module.forward)
            # logging.info(f"offloading {type(module)} activation to CPU")
        else:
            pass
            # logging.info(
            #     f"skipping because already offloaded. FWD = {digest_forward_name(module.forward)}"
            # )
    else:
        module.forward = _legokit_original_fwd[module]
        pass


class GaussianMeter:
    def __init__(self, window: int, name: str):
        self.window_metric = []
        self.window = window
        self.context = {}
        self.name = name
        self.verbose = False

    def set_name(self, name):
        self.name = name

    def add(self, metric) -> None:
        if self.window == len(self.window_metric):
            self.window_metric.pop(0)

        self.window_metric.append(metric)

    def find(self, metric) -> int:
        return self.window_metric.index(metric)

    def add_if_not_exist(self, metric) -> bool:
        if metric not in self.window_metric:
            self.add(metric)
            return True
        else:
            self.window_metric.index(metric)
            return False

    def replace_window_item(self, idx, metric) -> None:
        self.window_metric[idx] = metric

    def sample(self) -> float:
        mean = np.mean(self.window_metric)
        variance = np.var(self.window_metric)
        return np.random.normal(mean, variance)

    def add_context(self, k, v):
        if self.verbose:
            logging.info(f" {self.name}/{k}={v}")
        self.context[k] = v

    def purge_context(self, k):
        del self.context[k]

    def get_context(self, k):
        return self.context[k]

    def stats(self):
        return np.mean(self.window_metric), np.var(self.window_metric)

    def reset(self):
        self.context.clear()
        self.window_metric.clear()

    def mean_confidence_interval(self, confidence=0.9):
        if len(self.window_metric) < 2:
            return [self.window_metric[0]] * 3

        import numpy as np
        import scipy.stats

        a = 1.0 * np.array(self.window_metric)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        logging.info(f"[{self.name}] window = {self.window_metric}")
        return m, m - h, m + h

    def percentile(self, p):
        return np.percentile(self.window_metric, p)
