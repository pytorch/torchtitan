from typing import Dict, Optional, Tuple

import torch

from torch.autograd.graph import saved_tensors_hooks


HandleKey = Tuple[torch.device, torch.Tensor]


class Handle:
    def __init__(
        self,
        device_tensor: torch.Tensor,
        offload_stream: torch.cuda.Stream,
    ):
        if not torch.is_tensor(device_tensor):
            raise ValueError(f"Expects tensor but got {device_tensor}")
        self.device_tensor: Optional[torch.Tensor] = device_tensor
        self.cpu_tensor: Optional[torch.Tensor] = None
        self.offload_stream = offload_stream
        self.d2h_event: Optional[torch.cuda.Event] = None
        self.h2d_event: Optional[torch.cuda.Event] = None
        self.device: torch.device = device_tensor.device

    def copy_d2h_async(self) -> None:
        current_stream = torch.cuda.current_stream()
        self.offload_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.offload_stream):
            self.cpu_tensor = self.device_tensor.to(
                torch.device("cpu"), non_blocking=True
            )
            self.d2h_event = self.offload_stream.record_event()

    def copy_h2d_async(self) -> None:
        if self.device_tensor is not None:
            return
        assert self.cpu_tensor is not None
        self.device_tensor = torch.empty_like(self.cpu_tensor, device=self.device)
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            self.device_tensor.copy_(self.cpu_tensor, non_blocking=True)
            self.h2d_event = self.offload_stream.record_event()

    def wait_for_d2h(self):
        if self.d2h_event:
            torch.cuda.current_stream().wait_event(self.d2h_event)
        self.device_tensor = None

    def wait_for_h2d(self):
        if self.h2d_event:
            torch.cuda.current_stream().wait_event(self.h2d_event)
        self.cpu_tensor = None


class offload_to_cpu(saved_tensors_hooks):
    """
    This represents a saved tensors hooks context that offloads activations to
    CPU in forward and un-offloads them from CPU in backward.

    In forward, the D2H copy is always async. Device memory is freed when the
    user calls :meth:`wait_for_d2h`, which should be done after the compute
    with which to overlap has been issued.

    In backward, the H2D copy defaults to sync. However, the user may call
    :meth:`copy_h2d_async` to issue the H2D copy as async before the compute
    with which to overlap has been issued. When the activation is used in
    backward, we will wait for that H2D copy without user intervention.

    The D2H and H2D copies always used pinned memory, so the user should take
    care to ensure sufficient CPU RAM to be pinned. Otherwise the program can
    become slow or freeze. The first few iterations will be much slower due to
    repeated ``cudaHostAlloc`` calls to warmup the CPU caching allocator.
    """

    def __init__(self, offload_stream: torch.cuda.Stream):
        self.handle_key_to_handle: Dict[HandleKey, Handle] = {}
        self.offload_stream = offload_stream

        def pack_to_cpu(tensor: torch.Tensor):
            if tensor.device.type == "cpu":
                return (tensor.device, tensor)

            device_tensor = tensor
            del tensor
            # TODO: Need a way to decide whether to offload this tensor or not
            # that might need to be a function of the op constructing this
            # tensor, pipeline parallel rank, etc.
            if device_tensor.numel() < (14336 * 8192):  # (FFN dim * seq_len) for 8B
                return (device_tensor.device, device_tensor)

            handle = Handle(device_tensor, offload_stream)
            handle.copy_d2h_async()

            assert handle.cpu_tensor is not None
            handle_key = (device_tensor.device, handle.cpu_tensor)
            self.handle_key_to_handle[handle_key] = handle

            return handle_key

        def unpack_from_cpu(handle_key: HandleKey):
            device, tensor = handle_key
            if tensor.device == device:
                return tensor

            assert tensor.device == torch.device("cpu"), f"{tensor.device}"
            cpu_tensor = tensor
            del tensor

            handle = self.handle_key_to_handle.get(handle_key, None)
            if handle is None:
                raise RuntimeError(f"Handle missing for {handle_key}")

            handle.wait_for_h2d()
            if handle.device_tensor is not None:
                device_tensor = handle.device_tensor
                handle.device_tensor = None
                return device_tensor

            # Fallback to non-overlapped H2D copy
            device_tensor = cpu_tensor.to(device, non_blocking=True)
            assert handle.cpu_tensor is None
            return device_tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)

    def wait_for_d2h(self):
        for handle in self.handle_key_to_handle.values():
            handle.wait_for_d2h()

    def copy_h2d_async(self):
        # HACK: Sleeping for 1 ms before copy H2D helps avoid the no-overlap
        # issue for `reshard_after_forward=True` where AG copy-out's H2D copy
        # serializes after these H2D copies, preventing overlap.
        # self.offload_stream.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(self.offload_stream):
        #     from torch.testing._internal.common_utils import get_cycles_per_ms
        #     torch.cuda._sleep(int(get_cycles_per_ms()))
        for handle in self.handle_key_to_handle.values():
            handle.copy_h2d_async()

    def __enter__(self):
        super().__enter__()
        # Override this to return `self` so that the context can be saved like
        # with `offload_to_cpu(offload_stream) as ctx:`
        return self
