import torch
import logging
import random
from torch.nn import Module
from torch.autograd.graph import saved_tensors_hooks
from typing import NamedTuple
from collections import defaultdict


logger = logging.getLogger(__name__)


class PackInfo(NamedTuple):
    # Record an event in the offload stream for the default stream to wait on
    # before freeing the device tensor
    d2h_event: torch.cuda.Event
    # Keep a ref to the device tensor until the event has been waited on
    device_tensor: torch.Tensor


class UnpackInfo(NamedTuple):
    # Record an event during preallocation for the offload stream to wait on
    # before copying to the device tensor
    prealloc_event: torch.cuda.Event
    # Preallocate the device tensor memory so it can be allocated in the
    # default stream (instead of offload stream) to avoid fragmentation
    device_tensor: torch.Tensor


# TODO: Remove these from global namespace and register on modules. Consider
# using module state as identifier instead of int ID.
# Used or overlapping H2D/D2H copy with compute
offload_stream: torch.cuda.Stream = torch.cuda.Stream()
# Used for module ordering
module_id_to_module: dict[int, Module] = {}
next_module_id = 0
# Used in forward to keep device tensors alive through D2H copies
module_to_pack_infos: dict[Module, list[PackInfo]] = defaultdict(list)
# Appended to in forward and used in backward to know which CPU tensors will be
# copied H2D in backward to preallocate their device memory
module_to_cpu_tensors: dict[Module, list[torch.Tensor]] = defaultdict(list)
# Used in backward to preallocate device tensors in the default stream
cpu_tensor_to_unpack_info: dict[torch.Tensor, UnpackInfo] = {}


class activation_offload_with_overlap(saved_tensors_hooks):
    """
    In forward, we overlap the current module's D2H copies with the next
    module's forward compute.

    In backward, we overlap the current module's H2D copies with the previous
    module's backward compute.

    In backward, since we need to allocate new device memory for the H2D
    destinations, we can either do so in the offload stream or in the default
    stream. Naively, we may do so in the offload stream, but this fragments the
    memory pool since memory blocks are not shared across streams. As such, we
    instead choose to do so in the default stream. This requires preallocation
    and a CUDA event to ensure that the H2D copy does not start too early,
    using the default stream memory before it should.

    """

    def __init__(self, module: Module, offload_ratio: float = 1.0) -> None:
        global next_module_id

        module_id = next_module_id
        module_id_to_module[module_id] = module
        next_module_id += 1
        self.ignore_types = [torch.complex64, torch.int64]
        self.min_tensor_size_bytes = 1 * 1024 * 1024
        self.offload_ratio = max(0.0, min(1.0, offload_ratio))
        self.tensors_offloaded = 0
        self.tensors_kept_on_gpu = 0

        # logger.info(f"This is module {id(module):#x}, {module_id}.")

        def get_num_bytes_tensor(x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return x.element_size() * x.nelement() #x.element_size() * x._base_storage().nbytes()

        def pack_to_cpu(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
            if tensor.device.type == "cpu":
                # logger.info(f"")
                return (tensor.device, tensor)

            num_bytes = get_num_bytes_tensor(tensor)
            sizes = tensor.size()

            device_tensor = tensor  # rename for clarity
            del tensor

            # TODO: Insert optional policy for deciding whether to offload.
            # Migrate to be like non-reentrant activation checkpointing in the
            # future to reuse the selective activation checkpointing logic.
            if (device_tensor.numel() < self.min_tensor_size_bytes) or (device_tensor.dtype in self.ignore_types):
                # logger.info(f"Ignoring activation tensor of {num_bytes} bytes, size = {sizes}, dtype = {device_tensor.dtype}")
                return (device_tensor.device, device_tensor)

            should_offload = (self.tensors_offloaded / (self.tensors_offloaded + self.tensors_kept_on_gpu + 1) < self.offload_ratio)
            # should_offload = random.random() < self.offload_ratio
            if not should_offload:
                self.tensors_kept_on_gpu += 1
                return (device_tensor.device, device_tensor)

            current_stream = torch.cuda.current_stream()

            module_id_to_free = module_id - 1
            if module_id_to_free in module_id_to_module:
                # Have the first of module i to free all of module i-1
                # logger.info(f"Trying to free {module_id_to_free}...")
                module_to_free = module_id_to_module[module_id_to_free]
                self.free_packed_device_tensors(module_to_free)

            offload_stream.wait_stream(current_stream)
            with torch.cuda.stream(offload_stream):
                # logger.info(f"Copying activation tensor of {num_bytes} bytes, size = {sizes}, dtype = {device_tensor.dtype} to CPU...")
                cpu_tensor = device_tensor.to(torch.device("cpu"), non_blocking=True)
                # logger.info(f"Record d2h event.")
                d2h_event = offload_stream.record_event()
                self.tensors_offloaded += 1

            module_to_cpu_tensors[module].append(cpu_tensor)
            module_to_pack_infos[module].append(PackInfo(d2h_event, device_tensor))
            return (device_tensor.device, cpu_tensor)

        def unpack_from_cpu(packed) -> torch.Tensor:
            device, tensor = packed
            if tensor.device == device:
                return tensor
            assert tensor.device == torch.device("cpu"), f"{tensor.device}"

            cpu_tensor = tensor  # rename for clarity
            del tensor

            # Clear any existing refs from forward (this should only happen for
            # the last module)
            self.free_packed_device_tensors(module)

            current_stream = torch.cuda.current_stream()
            module_id_to_prealloc = module_id - 1

            if module_id_to_prealloc in module_id_to_module:
                module_to_prealloc = module_id_to_module[module_id_to_prealloc]
                if module_to_prealloc in module_to_cpu_tensors:
                    cpu_tensors = module_to_cpu_tensors[module_to_prealloc]
                    for _cpu_tensor in cpu_tensors:
                        cpu_tensor_to_unpack_info[_cpu_tensor] = UnpackInfo(
                            current_stream.record_event(),
                            torch.empty_like(_cpu_tensor, device=device),
                        )
                    del module_to_cpu_tensors[module_to_prealloc]

            if cpu_tensor in cpu_tensor_to_unpack_info:  # prefetched
                event, device_tensor = cpu_tensor_to_unpack_info[cpu_tensor]
                offload_stream.wait_event(event)
                del cpu_tensor_to_unpack_info[cpu_tensor]
            else:
                device_tensor = torch.empty_like(cpu_tensor, device=device)
                # Preallocate the rest of the 1st backward module
                for _cpu_tensor in module_to_cpu_tensors[module]:
                    if _cpu_tensor is cpu_tensor:
                        continue
                    cpu_tensor_to_unpack_info[_cpu_tensor] = UnpackInfo(
                        current_stream.record_event(),
                        torch.empty_like(_cpu_tensor, device=device),
                    )
                del module_to_cpu_tensors[module]
                offload_stream.wait_stream(current_stream)

            with torch.cuda.stream(offload_stream):
                device_tensor.copy_(cpu_tensor, non_blocking=True)
            current_stream.wait_stream(offload_stream)

            return device_tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)

    def free_packed_device_tensors(self, module: torch.nn.Module):
        if module in module_to_pack_infos:
            # logger.info(f"Trying to free packed device tensors from module {id(module):#x}")
            if infos := module_to_pack_infos[module]:
                # Make sure that the default stream does not reuse any of
                # the previous activation memory until the D2H finish
                torch.cuda.current_stream().wait_event(infos[-1].d2h_event)
            del module_to_pack_infos[module]

