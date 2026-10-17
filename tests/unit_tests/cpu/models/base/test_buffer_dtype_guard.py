# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest import mock

import spmd_types as spmd
import torch
import torch.nn as nn
from spmd_types import SpmdType

from torchtitan.models.base.module import Module, Sequential


class TestBufferDtypeGuard(unittest.TestCase):
    class WithBuffer(Module):
        _dtype_pinned_buffers = frozenset({"buf", "optional"})

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0, 1, 7))
            self.register_buffer("buf", torch.linspace(0, 1, 7), persistent=False)
            self.register_buffer("ordinary", torch.ones(7))
            self.register_buffer("optional", None)

        def reset_parameters(self):
            nn.init.ones_(self.weight)

        def _init_self_buffers(self, *, buffer_device=None):
            self.buf = torch.linspace(
                0, 1, 7, device=buffer_device, dtype=torch.float32
            )

    def test_rejected_cast_preserves_buffer(self):
        for cast in (
            lambda m: m.half(),
            lambda m: m.bfloat16(),
            lambda m: m.double(),
            lambda m: m.to(torch.bfloat16),
            lambda m: m.to("meta", torch.bfloat16),
            lambda m: m.type(torch.float16),
        ):
            for parent_type in (Sequential, nn.Sequential):
                with self.subTest(cast=cast, parent=parent_type):
                    child = self.WithBuffer()
                    original = child.buf
                    expected = original.clone()
                    with self.assertRaisesRegex(
                        ValueError, "before to_empty.*init_states"
                    ):
                        cast(parent_type(child))
                    self.assertIs(child.buf, original)
                    torch.testing.assert_close(child.buf, expected, rtol=0, atol=0)
                    # No rollback contract for parameters or earlier siblings.

    def test_meta_cast_materialize_initialize(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                with torch.device("meta"):
                    m = self.WithBuffer()
                m.to(dtype)
                self.assertEqual(m.buf.dtype, dtype)
                m.to_empty(device="cpu")
                m.init_states(buffer_device=torch.device("cpu"))
                self.assertEqual(m.weight.dtype, dtype)
                torch.testing.assert_close(m.buf, torch.linspace(0, 1, 7))

    def test_allowed_operations_and_one_call_per_tensor(self):
        m = self.WithBuffer()
        expected = m.buf.clone()
        m.float().cpu().to(memory_format=torch.contiguous_format)
        torch.testing.assert_close(m.buf, expected, rtol=0, atol=0)
        fn = mock.Mock(side_effect=lambda t: t.clone())
        m._apply(fn)
        self.assertEqual(fn.call_count, 3)
        m.to_empty(device="cpu")
        self.assertEqual(m.buf.dtype, torch.float32)
        m.to("meta")
        self.assertTrue(m.buf.is_meta)

    def test_unprotected_absent_and_aliased_buffers(self):
        m = self.WithBuffer()
        del m.buf
        m.half()
        self.assertEqual(m.ordinary.dtype, torch.float16)
        self.assertIsNone(m.optional)
        m = self.WithBuffer()
        # The unprotected name comes first, so named_buffers must not deduplicate.
        m.ordinary = m.buf
        m._buffers = {"ordinary": m.buf, "buf": m.buf}
        with self.assertRaises(ValueError):
            m.half()

    def test_complex_buffer_rejects_loss_of_imaginary_values(self):
        m = self.WithBuffer()
        m.buf = torch.complex(torch.ones(7), torch.arange(7, dtype=torch.float32))
        original = m.buf
        expected = original.clone()
        with self.assertRaises(ValueError):
            m.to(torch.bfloat16)
        self.assertIs(m.buf, original)
        torch.testing.assert_close(m.buf, expected, rtol=0, atol=0)
        # PyTorch's floating-only convenience casts leave complex tensors alone.
        m.half().bfloat16().float().double()
        torch.testing.assert_close(m.buf, expected, rtol=0, atol=0)

    def test_subclass_extends_protected_names(self):
        class Child(self.WithBuffer):
            _dtype_pinned_buffers = self.WithBuffer._dtype_pinned_buffers | frozenset(
                {"extra"}
            )

        m = Child()
        m.register_buffer("extra", torch.ones(7, dtype=torch.float64))
        # The float32 parent buffer is unchanged, so only the added pin rejects.
        with self.assertRaises(ValueError):
            m.float()
        self.assertEqual(m.extra.dtype, torch.float64)
        self.assertIn("buf", m._dtype_pinned_buffers)
        self.assertNotIn("extra", self.WithBuffer._dtype_pinned_buffers)

    def test_recurse_false(self):
        parent = Sequential(self.WithBuffer())
        parent._apply(lambda t: t.half(), recurse=False)
        self.assertEqual(parent[0].buf.dtype, torch.float32)
        with self.assertRaises(ValueError):
            parent.half()

    def test_integer_buffer_cast_semantics(self):
        m = self.WithBuffer()
        m.buf = torch.arange(7)
        m.half()
        self.assertEqual(m.buf.dtype, torch.int64)
        with self.assertRaises(ValueError):
            m.type(torch.float16)

    def test_checkpoint_assignment_is_not_a_cast(self):
        m = self.WithBuffer()
        m.register_buffer("buf", m.buf, persistent=True)
        m.load_state_dict(
            {name: tensor.half() for name, tensor in m.state_dict().items()},
            assign=True,
        )
        self.assertEqual(m.buf.dtype, torch.float16)
        m.cpu()
        with self.assertRaises(ValueError):
            m.float()
        # The guard pins the loaded dtype; initialization restores float32.
        m.init_states()
        self.assertEqual(m.buf.dtype, torch.float32)

    def test_spmd_annotations_preserved(self):
        m = self.WithBuffer()
        annotation = SpmdType({})
        spmd.assert_type(m.buf, annotation)
        m.to_empty(device="cpu")
        self.assertTrue(spmd.has_local_type(m.buf))
        m.init_states(buffer_device=torch.device("cpu"))
        self.assertTrue(spmd.has_local_type(m.buf))
        with self.assertRaises(ValueError):
            m.half()
        self.assertTrue(spmd.has_local_type(m.buf))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_moves_and_combined_cast(self):
        m = self.WithBuffer()
        expected = m.buf.clone()
        m.cuda().cpu()
        torch.testing.assert_close(m.buf, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            m.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
        torch.testing.assert_close(m.buf, expected, rtol=0, atol=0)
