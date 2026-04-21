from __future__ import annotations

import functools
import inspect

import torch


def _capture_key(args: tuple[object, ...]) -> tuple[object, ...]:
    key: list[object] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            key.append(
                (
                    arg.data_ptr(),
                    tuple(arg.size()),
                    tuple(arg.stride()),
                    str(arg.dtype),
                    str(arg.device),
                )
            )
        else:
            key.append(("obj", id(arg)))
    return tuple(key)


class _CudaGraphReplayState:
    def __init__(self) -> None:
        self.disabled = False
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs = None
        self.captured_key: tuple[object, ...] | None = None
        self.last_key: tuple[object, ...] | None = None
        self.seen_count = 0

    def run(self, module, orig_forward, args: tuple[object, ...]):
        if self.disabled or not torch.cuda.is_available():
            return orig_forward(module, *args)

        key = _capture_key(args)
        if self.graph is not None and key == self.captured_key:
            self.graph.replay()
            return self.outputs

        if key == self.last_key:
            self.seen_count += 1
        else:
            self.last_key = key
            self.seen_count = 1

        if self.graph is None and self.seen_count >= 2:
            try:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = orig_forward(module, *args)
                self.graph = graph
                self.outputs = outputs
                self.captured_key = key
                return outputs
            except Exception:
                self.disabled = True
                self.graph = None
                self.outputs = None
                self.captured_key = None
                return orig_forward(module, *args)

        return orig_forward(module, *args)


def maybe_wrap_forward(orig_forward):
    signature = inspect.signature(orig_forward)
    state_attr = "_llm_trainer_cuda_graph_state"

    @functools.wraps(orig_forward)
    def wrapped(self, *args, **kwargs):
        if kwargs:
            return orig_forward(self, *args, **kwargs)

        state = getattr(self, state_attr, None)
        if state is None:
            state = _CudaGraphReplayState()
            setattr(self, state_attr, state)

        return state.run(self, orig_forward, args)

    wrapped.__signature__ = signature
    return wrapped
