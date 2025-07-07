from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch



class TorchDeviceMixin(ABC):
    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def detach(self):
        raise NotImplementedError


class Batch(ABC):
    """Abstract class that defines the necessary methods any `Batch` implementation needs to implement."""

    pass


@dataclass
class DatasetBatch(Batch, TorchDeviceMixin):
    """A batch of samples and its targets. Used to batch train a model."""

    samples: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    batch_dim: int = 0

    def to(self, device: torch.device):
        self.samples = {k: v.to(device) for k, v in self.samples.items()}
        self.targets = {k: v.to(device) for k, v in self.targets.items()}

    def detach(self):
        self.targets = {k: v.detach() for k, v in self.targets.items()}
        self.samples = {k: v.detach() for k, v in self.samples.items()}

    @property
    def device(self) -> torch.device:
        key = list(self.samples.keys())[0]
        return self.samples[key].device

    def __len__(self) -> int:
        key = list(self.samples.keys())[0]
        return self.samples[key].shape[self.batch_dim]


@dataclass
class InferenceResultBatch(Batch, TorchDeviceMixin):
    """Stores targets and predictions of an entire batch."""

    targets: dict[str, torch.Tensor]
    predictions: dict[str, torch.Tensor]
    batch_dim: int = 0

    def to_cpu(self):
        self.to(device=torch.device("cpu"))

    @property
    def device(self) -> torch.device:
        key = list(self.targets.keys())[0]
        return self.targets[key].device

    def to(self, device: torch.device):
        self.predictions = {k: v.to(device) for k, v in self.predictions.items()}
        self.targets = {k: v.to(device) for k, v in self.targets.items()}

    def detach(self):
        self.targets = {k: v.detach() for k, v in self.targets.items()}
        self.predictions = {k: v.detach() for k, v in self.predictions.items()}

    def get_predictions(self, key: str) -> torch.Tensor:
        if key not in self.predictions:
            raise Exception(f"Key {key} not present in predictions!")
        return self.predictions[key]

    def get_targets(self, key: str) -> torch.Tensor:
        if key not in self.targets:
            raise Exception(f"Key {key} not present in targets!")
        return self.targets[key]

    def __len__(self) -> int:
        key = list(self.predictions.keys())[0]
        return self.predictions[key].shape[self.batch_dim]


@dataclass
class ResultItem:
    value: torch.Tensor
    decimal_places: Optional[int] = None


@dataclass
class EvaluationResultBatch(Batch):
    """Data class for storing the results of a single or multiple batches.
    Also entire epoch results are stored in here."""

    dataloader_tag: str
    num_train_steps_done: int
    losses: dict[str, ResultItem] = field(default_factory=dict)
    metrics: dict[str, ResultItem] = field(default_factory=dict)
    throughput_metrics: dict[str, ResultItem] = field(default_factory=dict)

    def __str__(self) -> str:
        def _round_result_item_dict(result_item_dict: dict[str, ResultItem]) -> dict[str, ResultItem]:
            rounded_result_item_dict = {}
            for k, item in result_item_dict.items():
                if item.decimal_places is not None:
                    rounded_result_item_dict[k] = round(item.value.mean().item(), item.decimal_places)
                else:
                    rounded_result_item_dict[k] = item.value.mean()
            return rounded_result_item_dict

        eval_str = f"Dataloader: {self.dataloader_tag} | "
        eval_str = f"step: {self.num_train_steps_done} | "

        eval_str += (
            " | ".join([f"{k}: {item}" for k, item in _round_result_item_dict(self.throughput_metrics).items()]) + " | "
        )
        eval_str += " | ".join([f"{k}: {item}" for k, item in _round_result_item_dict(self.losses).items()]) + " | "
        eval_str += " | ".join([f"{k}: {item}" for k, item in _round_result_item_dict(self.metrics).items()]) + " | "
        return eval_str
