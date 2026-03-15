# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    run_traced_module,
    trace_module,
)


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


class TrainStepModule(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args):
        *fwd_args, labels = args
        logits = self.model(*fwd_args)
        loss = self.loss_fn(logits, labels)
        # Must look up params in forward (not __init__) so that
        # _reparametrize_module's swapped parameters are captured during tracing.
        params = [p for _, p in self.model.named_parameters(remove_duplicate=False)]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)


def _get_params_and_buffers(mod):
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(device))
    return model


class SimpleMLP(nn.Module):
    def __init__(self, dim=64, hidden=128, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.embed(x))))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModule(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _make_mlp(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        return model, tokens, labels, get_loss

    def test_mlp_forward(self):
        model, tokens, labels, loss_fn = self._make_mlp()
        traced_result = trace_module(model, (tokens,))
        out_eager = model(tokens)
        pab = _get_params_and_buffers(model)
        wrapped = run_traced_module(traced_result, pab, (tokens,))
        self.assertTrue(torch.equal(out_eager, wrapped[0]))

    def test_mlp_train_step(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        train_step = TrainStepModule(model_ref, loss_fn)
        traced_result = trace_module(train_step, (tokens, labels))

        logits_ref = model_ref(tokens)
        loss_ref = loss_fn(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        train_step_copy = TrainStepModule(model_copy, loss_fn)
        pab = _get_params_and_buffers(train_step_copy)
        wrapped = run_traced_module(traced_result, pab, (tokens, labels))
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr, gt))

    def test_mlp_multistep_bitwise(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        train_step_ref = TrainStepModule(model_ref, loss_fn)
        train_step_copy = TrainStepModule(model_copy, loss_fn)
        traced_result = trace_module(train_step_ref, (tokens, labels))

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=self.LR)
        opt_copy = torch.optim.Adam(model_copy.parameters(), lr=self.LR)

        for step in range(1, self.NUM_STEPS + 1):
            logits_ref = model_ref(tokens)
            loss_ref = loss_fn(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            pab = _get_params_and_buffers(train_step_copy)
            wrapped = run_traced_module(traced_result, pab, (tokens, labels))
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_copy.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr),
                f"Step {step}: loss mismatch",
            )
            self.assertTrue(
                all(
                    torch.equal(gr, gt)
                    for gr, gt in zip(grads_ref, grads_tr, strict=True)
                ),
                f"Step {step}: grad mismatch",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceDTensor(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12357",
                world_size=1,
                rank=0,
            )
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        import torch.distributed as dist

        torch.use_deterministic_algorithms(False)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _distribute_params(self, model, mesh):
        from torch.distributed._tensor import distribute_tensor, Replicate

        for name, param in list(model.named_parameters()):
            dt = distribute_tensor(param, mesh, [Replicate()])
            param_parts = name.split(".")
            mod = model
            for part in param_parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, param_parts[-1], nn.Parameter(dt))

    def test_dtensor_forward(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        self._distribute_params(model, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])

        traced_result = trace_module(model, (tokens_dt,))
        has_subclass = any(
            layout.meta is not None for layout in traced_result.input_subclass_layouts
        )
        self.assertTrue(has_subclass)

        out_eager = model(tokens_dt)
        pab = _get_params_and_buffers(model)
        wrapped = run_traced_module(traced_result, pab, (tokens_dt,))
        self.assertTrue(
            torch.equal(out_eager.full_tensor(), wrapped[0].full_tensor())
        )

    def test_dtensor_train_step(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model_ref = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        self._distribute_params(model_ref, mesh)
        self._distribute_params(model_copy, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])
        labels_dt = DTensor.from_local(labels, mesh, [Replicate()])

        train_step = TrainStepModule(model_ref, get_loss)
        traced_result = trace_module(train_step, (tokens_dt, labels_dt))

        logits_ref = model_ref(tokens_dt)
        loss_ref = get_loss(logits_ref, labels_dt)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        train_step_copy = TrainStepModule(model_copy, get_loss)
        pab = _get_params_and_buffers(train_step_copy)
        wrapped = run_traced_module(traced_result, pab, (tokens_dt, labels_dt))
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(
            torch.equal(loss_ref.full_tensor(), loss_tr.full_tensor())
        )
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr.full_tensor(), gt.full_tensor()))


if __name__ == "__main__":
    unittest.main()
