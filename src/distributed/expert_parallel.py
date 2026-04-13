from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import (
    DeviceMesh,  # type: ignore
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle

from src.models.moe.utils import _permute, _unpermute

# ? a parallel style is an abstract base class that defines a single contract: a plan for how to parallelize a module
# ? a parallel style encodes theree things:
# ? 1. how to shard the module's parameters across a device mesh (partition_fn)
# ? 2. how to transform inputs before the module's forward runs (e.g., shard, replicate, or all-gather)
# ? 3. how to transform outputs after the module's forward runs (e.g., all-reduce, scatter, leave alone)
# ? it is originally designed for TP of nn.Linear layers


class BaseExpertParallel(ParallelStyle, ABC):
    @abstractmethod
    def _partition_fn(
        self, name: str, mod: nn.Module, device_mesh: DeviceMesh
    ) -> None: ...

    @abstractmethod
    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor: ...


class ExpertParallel(BaseExpertParallel):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        # ? shard all parameters at the start up
        for param_name, param in mod.named_parameters(
            recurse=False
        ):  # ? walk down the path, and replace the param with a sharded version
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )  # ? convert to a DTensor, sharded along the first dimension (expert dimension)
            # ? the nn.Parameter wrapper make the tensor to become a traninable parameter
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        # ? here, the routed input is (N * top_k, dim), expert-sorted tokens
        # ? and num_tokens_per_expert is (num_experts,), counting how many tokens routed to each expert on this device
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = (
            num_tokens_per_expert.shape[0] // ep_degree
        )  # ? how many experts are on this device

        with torch.no_grad():
            # * all-to-all is a collective communication primitive where every rank sends a piece of its data to every other rank
            # * as an example, all ranks send one element to every other rank
            # * Before:
            # * Rank 0: [a0, a1, a2, a3]
            # * Rank 1: [b0, b1, b2, b3]
            # * Rank 2: [c0, c1, c2, c3]
            # * Rank 3: [d0, d1, d2, d3]
            # * After:
            # * Rank 0: [a0, b0, c0, d0]
            # * Rank 1: [a1, b1, c1, d1]
            # * Rank 2: [a2, b2, c2, d2]
            # * Rank 3: [a3, b3, c3, d3]
            # * note: it is essentially a transpose of data matrix across the ranks, and it is very efficient for large data transfer across GPUs
            # * all to all signatures:
            # * all_to_all_single(
            # *     input,                    # the tensor to send, the tensor whose data gets distributed
            # *     output_split_sizes,       # how many elements we receive from each rank, a list of ints, length = world size. input_split_sizes[r] is the number of elements (along dim 0) from input to send to rank r. The sum must equal input.shape[0]
            # *     input_split_sizes,        # how many elements we send to each rank, output_split_sizes[r] is the number of elements this rank will receive from rank r. The sum determines the output tensor's dim-0 size.
            # *     group=None,  # the device mesh group to use for the communication
            # * )
            # * when None is used for output_split_sizes and input_split_sizes, it defaults to equal splits
            # ? in our cases:
            # ? input = num_tokens_per_expert
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,  # ? under the current set up, each rank should have the same num_tokens_per_expert, so it is defensive
                None,
                None,
                group=device_mesh.get_group(),
            )  # ? after this, each rank has the num_tokens_per_expert for all experts on its local device
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(  # ? the communication is asynchronous, so we need to wait for the result
                num_tokens_per_expert_group
            )
            # ? num_tokens_per_expert: [c0, c1, c2, c3, c4, c5, c6, c7]
            # ?    ↓ view(4, 2)
            # ? [[c0, c1],   ← row 0: counts for rank 0's experts
            # ? [c2, c3],   ← row 1: counts for rank 1's experts
            # ? [c4, c5],   ← row 2: counts for rank 2's experts
            # ? [c6, c7]]   ← row 3: counts for rank 3's experts
            # ?  [[c0, c1],         [c0+c1,
            # ? [c2, c3],    →     c2+c3,
            # ? [c4, c5],          c4+c5,
            # ? [c6, c7]]          c6+c7]
            input_splits = (  # ? sending
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            output_splits = (  # ? receiving
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        routed_input = all_to_all_single_autograd(  # ? reverse all to all communication. send the gradient back to the original rank
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # ? after communication, the tokens would look like this
        # ? Rows 0-19:    expert 4 tokens   ← from rank 0
        # ? Rows 20-29:   expert 5 tokens   ← from rank 0
        # ? Rows 30-49:   expert 4 tokens   ← from rank 1
        # ? Rows 50-59:   expert 5 tokens   ← from rank 1
        # ? Rows 60-79:   expert 4 tokens   ← from rank 2
        # ? Rows 80-89:   expert 5 tokens   ← from rank 2
        # ? Rows 90-109:  expert 4 tokens   ← from rank 3
        # ? Rows 110-119: expert 5 tokens   ← from rank 3
        # ? after _permute:
        # ? Rows 0-79:    all expert 4 tokens   (gathered from all four sources)
        # ? rows 0-19:   from rank 0
        # ? rows 20-39:  from rank 1
        # ? rows 40-59:  from rank 2
        # ? rows 60-79:  from rank 3
        # ? Rows 80-119:  all expert 5 tokens   (gathered from all four sources)
        # ? rows 80-89:   from rank 0
        # ? rows 90-99:   from rank 1
        # ? rows 100-109: from rank 2
        # ? rows 110-119: from rank 3
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(  # ? we are using the group mm kernel, which requires the input to be contiguously grouped by expert
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )  # ? reverse all to all communication. as if the ep never happened
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,  # ? run once at setyup to shard the parameters
            input_fn=self._token_dispatch,  # type: ignore  #? run as forward pre-hook on every call
            output_fn=self._token_combine,  # type: ignore  #? run as forward post-hook on every call
        )
