# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernel selection and mesh lookup."""

import importlib
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import context_parallel
from torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    HeadTailCPLoadBalancer,
    PTRRFlexAttentionCPLoadBalancer,
)
from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import (
    Attention,
    FlexInnerAttention,
    InnerAttention,
    MLAAttention,
    SlidingWindowFlexInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
    AllGatherKVSlidingWindowFlexInnerAttention,
    CPInnerAttention,
    UlyssesFlexInnerAttention,
    UlyssesInnerAttention,
    UlyssesSlidingWindowFlexInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_mla import (
    AllGatherKVMLAFlexAttention,
    UlyssesMLAAttention,
)
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import (
    attention_activation_placement,
    decoder_input_sharding,
    dense_activation_placement,
    mla_attention_local_spmd_sharding_config,
    mla_kv_projection_sharding_config,
)
from torchtitan.models.common.linear import Linear


def _mla_config_kwargs(
    inner_attention: InnerAttention.Config | None = None,
) -> dict:
    return {
        "wkv_b": Linear.Config(in_features=5, out_features=28),
        "packed_kv_head_dim": 7,
        "inner_attention": inner_attention or FlexInnerAttention.Config(),
    }


class _MockProjection(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.call = mock.Mock(return_value=output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x)


class TestMLAProjectionSharding(unittest.TestCase):
    def setUp(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        self.parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=2,
            tp=2,
            pp=1,
            ep=1,
            world_size=8,
            enable_sequence_parallel=False,
        )
        with mock.patch(
            "torchtitan.distributed.parallelism_context.device_type", "cpu"
        ):
            self.parallelism_context.build_mesh()

    def _build_attention(self, cp_placement):
        attention = MLAAttention.Config(
            wkv_b=Linear.Config(
                in_features=5,
                out_features=28,
                sharding_config=mla_kv_projection_sharding_config(cp=cp_placement),
            ),
            packed_kv_head_dim=7,
            inner_attention=FlexInnerAttention.Config(),
            sharding_config=mla_attention_local_spmd_sharding_config(),
        ).build()
        attention._parallelize(self.parallelism_context)
        return attention

    def test_projection_preserves_sharded_and_replicated_cp_inputs(self):
        for cp_placement in (spmd.S(0), spmd.R):
            with self.subTest(cp=cp_placement):
                attention = self._build_attention(cp_placement)
                latent = torch.randn(8, 5)
                with self.parallelism_context.activate_spmd(typechecking=True):
                    spmd.assert_type(
                        latent, dense_activation_placement(tp=spmd.R, cp=cp_placement)
                    )
                    projected = attention.project_mla_kv(latent)
                    spmd.assert_type(
                        projected, attention_activation_placement(cp=cp_placement)
                    )
                self.assertEqual(projected.shape, (8, 2, 7))

    def test_projection_rejects_a_tp_sharded_input(self):
        attention = self._build_attention(spmd.R)
        latent = torch.randn(8, 5)
        with self.parallelism_context.activate_spmd(typechecking=True):
            spmd.assert_type(
                latent, dense_activation_placement(tp=spmd.S(-1), cp=spmd.R)
            )
            with self.assertRaisesRegex(spmd.SpmdTypeError, "SPMD type mismatch"):
                attention.project_mla_kv(latent)

    def test_projection_rejects_a_mismatched_cp_input(self):
        for expected_cp, actual_cp in ((spmd.R, spmd.S(0)), (spmd.S(0), spmd.R)):
            with self.subTest(expected_cp=expected_cp, actual_cp=actual_cp):
                attention = self._build_attention(expected_cp)
                latent = torch.randn(8, 5)
                with self.parallelism_context.activate_spmd(typechecking=True):
                    spmd.assert_type(
                        latent, dense_activation_placement(tp=spmd.R, cp=actual_cp)
                    )
                    with self.assertRaisesRegex(
                        spmd.SpmdTypeError, "SPMD type mismatch"
                    ):
                        attention.project_mla_kv(latent)


class TestKernelSelection(unittest.TestCase):
    def test_mla_builders_set_projection_and_backend_sharding(self):
        for model, backends in (
            ("deepseek_v3", ("flex", "varlen")),
            ("kimi_k2_7", ("flex",)),  # QK clipping requires FlexAttention.
            ("kimi_k3", ("flex", "varlen")),
        ):
            build_model_config = importlib.import_module(
                f"torchtitan.models.{model}"
            ).build_model_config
            for enable_sp in (False, True):
                for backend in backends:
                    with self.subTest(
                        model=model, enable_sp=enable_sp, backend=backend
                    ):
                        config = build_model_config(
                            "debugmodel",
                            parallelism=ParallelismConfig(
                                tensor_parallel_degree=2,
                                enable_sequence_parallel=enable_sp,
                            ),
                            attn_backend=backend,
                            seq_len=128,
                        )
                        mla_configs = list(config.traverse(MLAAttention.Config))
                        self.assertTrue(mla_configs)
                        for _, mla_config, _, _ in mla_configs:
                            self.assertEqual(
                                mla_config.sharding_config,
                                mla_attention_local_spmd_sharding_config(),
                            )
                            self.assertEqual(
                                mla_config.wkv_b.sharding_config,
                                mla_kv_projection_sharding_config(cp=spmd.S(0)),
                            )
                            # The MLA backend already owns the local SPMD boundary.
                            self.assertIsNone(
                                mla_config.inner_attention.sharding_config
                            )

    def test_cp_kernel_is_a_flex_kernel(self):
        config = AllGatherKVFlexInnerAttention.Config()
        kernel = AllGatherKVFlexInnerAttention(config)
        self.assertIsInstance(config, CPInnerAttention.Config)
        self.assertIsInstance(config, InnerAttention.Config)
        self.assertIsInstance(config, FlexInnerAttention.Config)
        self.assertIsInstance(kernel, CPInnerAttention)
        self.assertIsInstance(kernel, InnerAttention)
        self.assertIsInstance(kernel, FlexInnerAttention)

    def test_mla_cp_kernel_inherits_inner_attention_and_cp_capability(self):
        config = AllGatherKVMLAFlexAttention.Config(**_mla_config_kwargs())
        kernel = AllGatherKVMLAFlexAttention(config)
        self.assertIsInstance(config, CPInnerAttention.Config)
        self.assertIsInstance(kernel, CPInnerAttention)
        self.assertIsInstance(config, InnerAttention.Config)
        self.assertIsInstance(kernel, InnerAttention)
        self.assertIsInstance(kernel, MLAAttention)
        self.assertIs(
            kernel.prepare_cp_metadata,
            AllGatherKVFlexInnerAttention.prepare_cp_metadata,
        )
        self.assertIsInstance(kernel.inner_attention, FlexInnerAttention)
        self.assertNotIsInstance(kernel, AllGatherKVFlexInnerAttention)

    def test_cp_kernel_inherits_flex_fields(self):
        config = AllGatherKVFlexInnerAttention.Config(block_size=256)
        self.assertEqual(config.block_size, 256)

    def test_sliding_cp_kernel_is_a_sliding_flex_kernel(self):
        config = AllGatherKVSlidingWindowFlexInnerAttention.Config(window_size=128)
        self.assertIsInstance(config, CPInnerAttention.Config)
        self.assertIsInstance(config, SlidingWindowFlexInnerAttention.Config)

    def test_cp_kernel_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("allgather_cp_flex")

    def test_plain_flex_is_not_a_cp_kernel(self):
        self.assertNotIsInstance(get_attention_config("flex"), CPInnerAttention.Config)

    def test_all_gather_backends_share_metadata_preparation(self):
        self.assertIs(
            AllGatherKVSlidingWindowFlexInnerAttention.prepare_cp_metadata,
            AllGatherKVFlexInnerAttention.prepare_cp_metadata,
        )
        self.assertIs(
            AllGatherKVMLAFlexAttention.prepare_cp_metadata,
            AllGatherKVFlexInnerAttention.prepare_cp_metadata,
        )

    def test_cp_configs_build_wrapped_backend_metadata(self):
        positions = torch.arange(8)
        full_metadata = AllGatherKVFlexInnerAttention.Config(
            block_size=4
        ).build_attention_metadata(positions)
        sliding_metadata = AllGatherKVSlidingWindowFlexInnerAttention.Config(
            block_size=4,
            window_size=4,
        ).build_attention_metadata(positions)
        varlen_metadata = UlyssesVarlenInnerAttention.Config().build_attention_metadata(
            positions
        )

        self.assertIsInstance(full_metadata, BlockMask)
        self.assertIsInstance(sliding_metadata, BlockMask)
        self.assertIsInstance(varlen_metadata, VarlenAttentionMetadata)


class TestCpMlaMetadata(unittest.TestCase):
    def test_full_attention_follows_wrapped_kernel(self):
        for backend in (MLAAttention, AllGatherKVMLAFlexAttention, UlyssesMLAAttention):
            for inner_attention, expected in (
                (FlexInnerAttention.Config(), True),
                (SlidingWindowFlexInnerAttention.Config(window_size=4), False),
                (VarlenInnerAttention.Config(), True),
                (VarlenInnerAttention.Config(window_size=(4, 0)), False),
            ):
                if backend is AllGatherKVMLAFlexAttention and isinstance(
                    inner_attention, VarlenInnerAttention.Config
                ):
                    continue
                with self.subTest(backend=backend, inner_attention=inner_attention):
                    config = backend.Config(**_mla_config_kwargs(inner_attention))
                    self.assertEqual(config.is_full_attention, expected)

    def test_varlen_metadata_preserves_document_options(self):
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], dtype=torch.int32)
        padding_mask = torch.tensor([False] * 6 + [True] * 2)
        inner_attention = VarlenInnerAttention.Config()
        expected = inner_attention.build_attention_metadata(
            positions,
            padding_mask=padding_mask,
            max_num_documents=4,
            max_context_length=16,
        )
        for backend in (MLAAttention, UlyssesMLAAttention):
            with self.subTest(backend=backend):
                config = backend.Config(**_mla_config_kwargs(inner_attention))
                actual = config.build_attention_metadata(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=4,
                    max_context_length=16,
                )
                self.assertIsInstance(actual, VarlenAttentionMetadata)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @staticmethod
    def _get_attention_metadata(config):
        attention = Attention.Config(n_heads=4, inner_attention=config)
        decoder = SimpleNamespace(config=SimpleNamespace(layers=[attention]))
        return Decoder._get_attention_metadata(cast(Decoder, decoder), torch.arange(8))

    def test_all_gather_uses_cp_backend_key(self):
        config = AllGatherKVMLAFlexAttention.Config(
            **_mla_config_kwargs(FlexInnerAttention.Config(block_size=4))
        )

        attention_metadata = self._get_attention_metadata(config)

        self.assertEqual({AllGatherKVMLAFlexAttention}, set(attention_metadata))
        self.assertIsInstance(
            attention_metadata[AllGatherKVMLAFlexAttention], BlockMask
        )

    def test_ulysses_mla_supports_varlen_metadata(self):
        config = UlyssesMLAAttention.Config(
            **_mla_config_kwargs(VarlenInnerAttention.Config())
        )

        attention_metadata = self._get_attention_metadata(config)

        self.assertEqual({UlyssesMLAAttention}, set(attention_metadata))
        self.assertIsInstance(
            attention_metadata[UlyssesMLAAttention], VarlenAttentionMetadata
        )


class TestFluxCpSharding(unittest.TestCase):
    def test_flux_input_sharding_uses_sequence_dim_one(self):
        from torchtitan.models.flux.sharding import flux_input_sharding

        input_shardings = flux_input_sharding()

        self.assertEqual(
            set(input_shardings),
            {"img", "img_ids", "txt", "txt_ids", "target"},
        )
        for sharding in input_shardings.values():
            cp = sharding.local_type[MeshAxisName.CP]
            self.assertIsInstance(cp, spmd.Shard)
            self.assertEqual(cp.dim, 1)


class TestDecoderCpSharding(unittest.TestCase):
    def test_dispatches_linear_cp_metadata_and_leaves_non_cp_metadata_unchanged(self):
        from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
        from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA

        for backend in (ContextParallelInnerGDN, ContextParallelInnerKDA):
            with self.subTest(backend=backend):
                global_metadata, local_metadata = object(), object()
                input_dict = {
                    "attention_metadata": {
                        backend: global_metadata,
                        FlexInnerAttention: global_metadata,
                        MLAAttention: global_metadata,
                    }
                }
                with mock.patch.object(
                    backend, "prepare_cp_metadata", return_value=local_metadata
                ) as prepare, mock.patch.object(
                    context_parallel, "shard_tensors", return_value=input_dict
                ):
                    Decoder._cp_shard(
                        mock.Mock(spec=Decoder),
                        input_dict,
                        input_shardings={},
                        parallelism_context=mock.Mock(),
                        parallelism=ParallelismConfig(
                            context_parallel_degree=2,
                            context_parallel_load_balancer=None,
                        ),
                    )
                prepare.assert_called_once_with(global_metadata, permutation=None)
                metadata = input_dict["attention_metadata"]
                self.assertIs(metadata[backend], local_metadata)
                self.assertIs(metadata[FlexInnerAttention], global_metadata)
                self.assertIs(metadata[MLAAttention], global_metadata)

    def _assert_selected_ptrr_metadata(
        self,
        *,
        attention_metadata,
        expected_metadata,
        sliding_type=AllGatherKVSlidingWindowFlexInnerAttention,
        full_type=AllGatherKVFlexInnerAttention,
        full_config=None,
    ):
        sliding_attention = mock.Mock(
            spec=Attention.Config,
            inner_attention=sliding_type.Config(window_size=128),
        )
        layers = [
            SimpleNamespace(attention=None),
            SimpleNamespace(attention=sliding_attention),
        ]
        full_attention = None
        if full_type in attention_metadata:
            full_attention = mock.Mock(
                spec=Attention.Config,
                inner_attention=(
                    full_config if full_config is not None else full_type.Config()
                ),
            )
            layers.append(SimpleNamespace(attention=full_attention))
        model = SimpleNamespace(
            config=SimpleNamespace(
                first_full_attention=full_attention,
                layers=layers,
            )
        )
        load_balancer = mock.Mock()
        load_balancer.generate_permutation.return_value = None
        load_balancer_config = PTRRFlexAttentionCPLoadBalancer.Config()
        parallelism = SimpleNamespace(
            context_parallel_load_balancer=load_balancer_config
        )
        input_dict = {
            "input": torch.arange(8),
            "attention_metadata": attention_metadata,
        }

        with mock.patch.object(
            PTRRFlexAttentionCPLoadBalancer.Config,
            "build",
            return_value=load_balancer,
        ) as build_load_balancer, mock.patch.object(
            context_parallel,
            "get_cp_input_seq_len",
            return_value=8,
        ), mock.patch.object(
            context_parallel,
            "shard_tensors",
            return_value=input_dict,
        ), mock.patch.object(
            full_type,
            "prepare_cp_metadata",
            side_effect=lambda metadata, **_kwargs: metadata,
        ), mock.patch.object(
            AllGatherKVSlidingWindowFlexInnerAttention,
            "prepare_cp_metadata",
            side_effect=lambda metadata, **_kwargs: metadata,
        ):
            Decoder._cp_shard(
                cast(Any, model),
                input_dict,
                input_shardings=decoder_input_sharding(),
                parallelism_context=cast(Any, SimpleNamespace()),
                parallelism=cast(Any, parallelism),
            )

        build_load_balancer.assert_called_once_with(
            seq_len=8,
            attention_metadata=expected_metadata,
        )

    def test_no_full_attention_disables_load_balancing(self):
        from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN
        from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA

        model = SimpleNamespace(config=SimpleNamespace(first_full_attention=None))
        for inner_attention in (
            AllGatherKVSlidingWindowFlexInnerAttention,
            ContextParallelInnerGDN,
            ContextParallelInnerKDA,
        ):
            for load_balancer_type in (
                HeadTailCPLoadBalancer,
                PTRRFlexAttentionCPLoadBalancer,
            ):
                with self.subTest(
                    attention=inner_attention.__name__,
                    load_balancer=load_balancer_type.__name__,
                ):
                    metadata = object()
                    input_dict = {
                        "input": torch.arange(8),
                        "attention_metadata": {inner_attention: metadata},
                    }
                    parallelism = SimpleNamespace(
                        context_parallel_load_balancer=load_balancer_type.Config()
                    )
                    with mock.patch.object(
                        load_balancer_type.Config, "build"
                    ) as build_load_balancer, mock.patch.object(
                        inner_attention,
                        "prepare_cp_metadata",
                        return_value=metadata,
                    ) as prepare_metadata, mock.patch.object(
                        context_parallel, "shard_tensors", return_value=input_dict
                    ) as shard_tensors, mock.patch(
                        "torchtitan.models.common.decoder.warn_once"
                    ) as warn:
                        Decoder._cp_shard(
                            cast(Any, model),
                            input_dict,
                            input_shardings=decoder_input_sharding(),
                            parallelism_context=cast(Any, SimpleNamespace()),
                            parallelism=cast(Any, parallelism),
                        )

                    build_load_balancer.assert_not_called()
                    prepare_metadata.assert_called_once_with(metadata, permutation=None)
                    self.assertIsNone(shard_tensors.call_args.kwargs["permutation"])
                    warn.assert_called_once()
                    self.assertIn("no full attention", warn.call_args.args[1])

    def test_ptrr_prefers_full_metadata(self):
        full_metadata = object.__new__(BlockMask)
        sliding_metadata = object.__new__(BlockMask)

        self._assert_selected_ptrr_metadata(
            attention_metadata={
                AllGatherKVFlexInnerAttention: full_metadata,
                AllGatherKVSlidingWindowFlexInnerAttention: sliding_metadata,
            },
            expected_metadata=full_metadata,
        )

    def test_ptrr_prefers_ulysses_full_metadata(self):
        full_metadata = object.__new__(BlockMask)
        sliding_metadata = object.__new__(BlockMask)
        self._assert_selected_ptrr_metadata(
            attention_metadata={
                UlyssesSlidingWindowFlexInnerAttention: sliding_metadata,
                UlyssesFlexInnerAttention: full_metadata,
            },
            expected_metadata=full_metadata,
            sliding_type=UlyssesSlidingWindowFlexInnerAttention,
            full_type=UlyssesFlexInnerAttention,
        )

    def test_ptrr_prefers_full_mla_metadata_over_sliding_metadata(self):
        full_metadata = object.__new__(BlockMask)
        sliding_metadata = object.__new__(BlockMask)
        self._assert_selected_ptrr_metadata(
            attention_metadata={
                AllGatherKVSlidingWindowFlexInnerAttention: sliding_metadata,
                AllGatherKVMLAFlexAttention: full_metadata,
            },
            expected_metadata=full_metadata,
            full_type=AllGatherKVMLAFlexAttention,
            full_config=AllGatherKVMLAFlexAttention.Config(**_mla_config_kwargs()),
        )

    def test_config_builds_selected_load_balancer(self):
        input_T = torch.arange(8)
        block_mask = object.__new__(BlockMask)
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")
        cases = (
            (HeadTailCPLoadBalancer, None),
            (
                PTRRFlexAttentionCPLoadBalancer,
                block_mask,
            ),
        )
        with mock.patch(
            "torchtitan.distributed.context_parallel._HeadTailLoadBalancer"
        ), mock.patch(
            "torchtitan.distributed.context_parallel._PTRRLoadBalancer"
        ), mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ):
            for load_balancer_type, attention_metadata in cases:
                with self.subTest(load_balancer_type=load_balancer_type.__name__):
                    load_balancer = load_balancer_type.Config().build(
                        seq_len=input_T.shape[0],
                        attention_metadata=attention_metadata,
                    )
                    self.assertIs(type(load_balancer), load_balancer_type)

    def test_base_load_balancer_config_is_not_buildable(self):
        with self.assertRaises(NotImplementedError):
            ContextParallelLoadBalancer.Config().build()

    def test_contiguous_token_fragments(self):
        self.assertEqual(
            context_parallel.get_token_fragments(
                8,
                cp_size=2,
                permutation=None,
            ),
            [[(0, 4)], [(4, 8)]],
        )

    def test_headtail_token_fragments(self):
        self.assertEqual(
            context_parallel.get_token_fragments(
                8,
                cp_size=2,
                permutation=torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5]]),
            ),
            [[(0, 2), (6, 8)], [(2, 6)]],
        )

    def test_token_fragments_reject_uneven_partition(self):
        with self.assertRaisesRegex(ValueError, "divisible by the CP degree"):
            context_parallel.get_token_fragments(
                7,
                cp_size=2,
                permutation=None,
            )

    def test_no_shardable_inputs_is_a_noop(self):
        batch = {"attention_metadata": object()}
        result = context_parallel.shard_tensors(
            batch,
            input_shardings={},
            permutation=None,
        )

        self.assertIs(result, batch)

    def test_sharded_sequence_length_must_be_divisible_by_cp_size(self):
        cp_group = SimpleNamespace(size=lambda: 2)
        with mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), self.assertRaisesRegex(ValueError, r"sequence length \(5\)"):
            context_parallel.shard_tensors(
                {"input": torch.arange(5)},
                input_shardings=decoder_input_sharding(),
                permutation=None,
            )

    def test_common_input_sharding_does_not_modify_metadata(self):
        input_T = torch.arange(8)
        labels_T = torch.arange(8)
        attention_metadata = object.__new__(BlockMask)
        batch = {
            "input": input_T,
            "labels": labels_T,
            "attention_metadata": attention_metadata,
        }
        sharded_input_T = input_T[:4]
        sharded_labels_T = labels_T[:4]
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")
        permutation = torch.arange(8).unsqueeze(0)
        torch_load_balancer = mock.Mock()
        torch_load_balancer._generate_indices.return_value = permutation

        with mock.patch(
            "torchtitan.distributed.context_parallel._HeadTailLoadBalancer",
            return_value=torch_load_balancer,
        ) as create_load_balancer, mock.patch(
            "torchtitan.distributed.context_parallel.spmd.shard",
            side_effect=(sharded_input_T, sharded_labels_T),
        ) as shard_input, mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ):
            load_balancer = HeadTailCPLoadBalancer.Config().build(
                seq_len=input_T.shape[0],
                attention_metadata=attention_metadata,
            )
            assert isinstance(load_balancer, ContextParallelLoadBalancer)
            permutation = load_balancer.generate_permutation()
            result = context_parallel.shard_tensors(
                batch,
                input_shardings=decoder_input_sharding(),
                permutation=permutation,
            )

        self.assertIs(result, batch)
        self.assertIs(result["input"], sharded_input_T)
        self.assertIs(result["labels"], sharded_labels_T)
        self.assertIs(result["attention_metadata"], attention_metadata)
        create_load_balancer.assert_called_once_with(8, 2, "cuda")
        self.assertEqual(shard_input.call_count, 2)
        self.assertIs(shard_input.call_args_list[0].args[1], cp_group)

    def test_headtail_uses_the_first_named_cp_input(self):
        tokens_BS = torch.arange(8).view(1, 8)
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")

        with mock.patch(
            "torchtitan.distributed.context_parallel._HeadTailLoadBalancer"
        ) as create_load_balancer, mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ):
            load_balancer = HeadTailCPLoadBalancer.Config().build(
                seq_len=tokens_BS.shape[1],
                attention_metadata=None,
            )
            load_balancer.generate_permutation()

        create_load_balancer.assert_called_once_with(8, 2, "cuda")

    def test_headtail_requires_two_chunks_per_cp_rank(self):
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")

        with mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ), self.assertRaisesRegex(
            ValueError, r"divisible by 2 \* CP size \(4\)"
        ):
            HeadTailCPLoadBalancer.Config().build(
                seq_len=6,
                attention_metadata=None,
            )

    def test_ptrr_is_rebuilt_from_each_batch_mask(self):
        input_T = torch.arange(8)
        first_mask = object.__new__(BlockMask)
        second_mask = object.__new__(BlockMask)
        cp_group = SimpleNamespace(size=lambda: 2)
        config = PTRRFlexAttentionCPLoadBalancer.Config()

        with mock.patch(
            "torchtitan.distributed.context_parallel._PTRRLoadBalancer"
        ) as create_load_balancer, mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ):
            first_load_balancer = config.build(
                seq_len=input_T.shape[0],
                attention_metadata=first_mask,
            )
            first_load_balancer.generate_permutation()
            second_load_balancer = config.build(
                seq_len=input_T.shape[0],
                attention_metadata=second_mask,
            )
            second_load_balancer.generate_permutation()

        self.assertEqual(
            create_load_balancer.call_args_list,
            [mock.call(first_mask, 2), mock.call(second_mask, 2)],
        )


class _FakeMesh:
    def __init__(self, cp_size: int | None):
        self.mesh_dim_names = ("dp", "tp") if cp_size is None else ("dp", "cp", "tp")
        self._cp_size = cp_size

    def get_group(self, axis):
        assert axis == "cp"
        return SimpleNamespace(size=lambda: self._cp_size)


def _in_mesh(cp_size):
    return mock.patch(
        "torchtitan.distributed.spmd_types.current_spmd_mesh",
        return_value=_FakeMesh(cp_size),
    )


class TestCpGroup(unittest.TestCase):
    """CP inner attention requires a multi-rank CP group."""

    @staticmethod
    def _kernel():
        return AllGatherKVFlexInnerAttention(AllGatherKVFlexInnerAttention.Config())

    def test_cp_axis_above_one_yields_its_group(self):
        with _in_mesh(8):
            group = spmd_mesh_group(MeshAxisName.CP)
            assert group is not None
            self.assertEqual(group.size(), 8)

    def test_no_mesh_context_returns_none(self):
        with mock.patch(
            "torchtitan.distributed.spmd_types.current_spmd_mesh", return_value=None
        ):
            self.assertIsNone(spmd_mesh_group(MeshAxisName.CP))

    def test_degree_one_returns_none(self):
        with _in_mesh(1):
            self.assertIsNone(spmd_mesh_group(MeshAxisName.CP))

    def test_mesh_without_a_cp_axis_returns_none(self):
        with _in_mesh(None):
            self.assertIsNone(spmd_mesh_group(MeshAxisName.CP))

    def test_forward_without_a_cp_group_is_an_error(self):
        num_tokens, heads, head_dim = 8, 2, 16
        q, k, v = (torch.randn(num_tokens, heads, head_dim) for _ in range(3))
        with _in_mesh(1), self.assertRaisesRegex(
            RuntimeError, "active multi-rank CP mesh axis"
        ):
            self._kernel().forward(q, k, v)

    def test_cp_attention_holds_no_mesh_state(self):
        self.assertNotIn("cp_group", CPInnerAttention.__dict__)


class TestAllGather(unittest.TestCase):
    def test_gathers_k_and_v_over_the_cp_group(self):
        num_tokens, heads, head_dim = 8, 2, 16
        q, k, v = (torch.randn(num_tokens, heads, head_dim) for _ in range(3))
        calls = []

        def record(x, group, *, src, dst, backward_options):
            calls.append((x, group, src, dst, backward_options))
            return x

        with _in_mesh(8), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, *a, **kw: q
        ):
            AllGatherKVFlexInnerAttention(
                AllGatherKVFlexInnerAttention.Config()
            ).forward(q, k, v)

        self.assertEqual(2, len(calls))
        self.assertIs(k, calls[0][0])
        self.assertIs(v, calls[1][0])
        for _, group, src, dst, backward_options in calls:
            self.assertEqual(8, group.size())
            self.assertEqual(spmd.S(0), src)
            self.assertEqual(spmd.R, dst)
            self.assertEqual({"op_dtype": torch.float32}, backward_options)

    @staticmethod
    def _reduce_dtypes(config):
        """Reduction dtype the kernel asks for, once per gathered tensor."""
        seen = []

        def record(x, group, *, src, dst, backward_options):
            seen.append(backward_options["op_dtype"])
            return x

        q, k, v = (torch.randn(8, 2, 16, dtype=torch.bfloat16) for _ in range(3))
        with _in_mesh(8), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, *a, **kw: q
        ):
            AllGatherKVFlexInnerAttention(config).forward(q, k, v)
        return seen

    def test_reduces_in_float32_by_default(self):
        config = AllGatherKVFlexInnerAttention.Config()
        self.assertEqual([torch.float32] * 2, self._reduce_dtypes(config))

    def test_reduce_dtype_can_use_bfloat16(self):
        config = AllGatherKVFlexInnerAttention.Config(reduce_dtype="bfloat16")
        self.assertEqual([torch.bfloat16] * 2, self._reduce_dtypes(config))


class TestAllGatherCollective(unittest.TestCase):
    """Exercise the real collective and its backward.

    A single-rank group is enough: the reduction dtype is validated first.
    """

    @classmethod
    def setUpClass(cls):
        cls._owns_pg = not dist.is_initialized()
        if cls._owns_pg:
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12362",
                world_size=1,
                rank=0,
            )

    @classmethod
    def tearDownClass(cls):
        if cls._owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    def _gather_and_backward(self, dtype):
        """Pair each of K and V with the gradient the gather returns to it."""
        kernel = AllGatherKVFlexInnerAttention(AllGatherKVFlexInnerAttention.Config())
        q, k, v = (
            torch.randn(4, 2, 8, dtype=dtype, requires_grad=True) for _ in range(3)
        )
        with mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=dist.group.WORLD,
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, k, v, **kw: k + v
        ):
            kernel.forward(q, k, v).float().sum().backward()
        k_grad, v_grad = k.grad, v.grad
        assert k_grad is not None and v_grad is not None
        return ((k, k_grad), (v, v_grad))

    def test_bfloat16_kv_reach_the_reducing_backward(self):
        for tensor, grad in self._gather_and_backward(torch.bfloat16):
            self.assertEqual(torch.bfloat16, grad.dtype)
            self.assertEqual(tensor.shape, grad.shape)

    def test_float32_kv_reach_the_reducing_backward(self):
        for _, grad in self._gather_and_backward(torch.float32):
            self.assertEqual(torch.float32, grad.dtype)


class TestUlysses(unittest.TestCase):
    def test_shared_base_is_a_cp_attention(self):
        self.assertTrue(issubclass(UlyssesInnerAttention, CPInnerAttention))
        self.assertTrue(
            issubclass(UlyssesInnerAttention.Config, CPInnerAttention.Config)
        )

    def test_is_still_a_flex_kernel(self):
        config = UlyssesFlexInnerAttention.Config()
        self.assertIsInstance(config, UlyssesInnerAttention.Config)
        self.assertIsInstance(config, FlexInnerAttention.Config)

    def test_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("ulysses_cp_flex")

    def test_reshards_sequence_to_heads_and_back(self):
        q, k, v = (torch.randn(8, 4, 16) for _ in range(3))
        calls = []

        def record(x, group, *, src, dst):
            calls.append((x, group, src, dst))
            return x

        kernel = UlyssesFlexInnerAttention(UlyssesFlexInnerAttention.Config())
        with _in_mesh(2), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, *a, **kw: q
        ):
            kernel.forward(q, k, v)

        self.assertEqual(4, len(calls))
        for _, group, src, dst in calls[:3]:
            self.assertEqual(2, group.size())
            self.assertEqual(spmd.S(0), src)
            self.assertEqual(spmd.S(1), dst)
        _, group, src, dst = calls[3]
        self.assertEqual(2, group.size())
        self.assertEqual(spmd.S(1), src)
        self.assertEqual(spmd.S(0), dst)


class TestMLAContextParallelAttention(unittest.TestCase):
    def test_mla_flex_materializes_per_head_kv_at_kernel_boundary(self):
        q_THK = torch.randn(8, 4, 6)
        kv_c_normed_TL = torch.randn(8, 5)
        kv_THP = torch.randn(8, 4, 7)
        k_shared_TR = torch.randn(8, 2)
        wkv_b = _MockProjection(kv_THP.flatten(-2))
        expected_k_THK = torch.cat(
            (kv_THP[..., :4], k_shared_TR.unsqueeze(1).expand(-1, 4, -1)), dim=-1
        )
        expected_v_THV = kv_THP[..., 4:]
        out_THV = torch.randn(8, 4, 3)
        attention_mask = object()
        kernel = MLAAttention(MLAAttention.Config(**_mla_config_kwargs()))
        kernel.wkv_b = wkv_b

        with mock.patch.object(
            FlexInnerAttention,
            "forward",
            autospec=True,
            return_value=out_THV,
        ) as flex_forward:
            result = kernel.forward(
                q_THK,
                kv_c_normed_TL,
                k_shared_TR,
                attention_metadata=attention_mask,
            )

        self.assertIs(result, out_THV)
        wkv_b.call.assert_called_once_with(kv_c_normed_TL)
        args = flex_forward.call_args.args
        self.assertIs(args[0], kernel.inner_attention)
        self.assertIs(args[1], q_THK)
        torch.testing.assert_close(args[2], expected_k_THK)
        torch.testing.assert_close(args[3], expected_v_THV)
        self.assertIs(
            flex_forward.call_args.kwargs["attention_metadata"], attention_mask
        )

    def test_mla_varlen_materializes_per_head_kv_at_kernel_boundary(self):
        q_THK = torch.randn(8, 4, 6)
        kv_c_normed_TL = torch.randn(8, 5)
        kv_THP = torch.randn(8, 4, 7)
        k_shared_TR = torch.randn(8, 2)
        wkv_b = _MockProjection(kv_THP.flatten(-2))
        expected_k_THK = torch.cat(
            (kv_THP[..., :4], k_shared_TR.unsqueeze(1).expand(-1, 4, -1)), dim=-1
        )
        expected_v_THV = kv_THP[..., 4:]
        out_THV = torch.randn(8, 4, 3)
        attention_mask = object()
        kernel = MLAAttention(
            MLAAttention.Config(**_mla_config_kwargs(VarlenInnerAttention.Config()))
        )
        kernel.wkv_b = wkv_b

        with mock.patch.object(
            VarlenInnerAttention,
            "forward",
            autospec=True,
            return_value=out_THV,
        ) as varlen_forward:
            result = kernel.forward(
                q_THK,
                kv_c_normed_TL,
                k_shared_TR,
                attention_metadata=attention_mask,
            )

        self.assertIs(result, out_THV)
        wkv_b.call.assert_called_once_with(kv_c_normed_TL)
        args = varlen_forward.call_args.args
        self.assertIs(args[0], kernel.inner_attention)
        self.assertIs(args[1], q_THK)
        torch.testing.assert_close(args[2], expected_k_THK)
        torch.testing.assert_close(args[3], expected_v_THV)
        self.assertIs(
            varlen_forward.call_args.kwargs["attention_metadata"], attention_mask
        )

    def test_all_gather_communicates_compressed_mla_inputs(self):
        q_THK = torch.randn(8, 4, 6)
        kv_c_normed_TL = torch.randn(8, 5)
        k_shared_TR = torch.randn(8, 2)
        gathered_kv_c_normed_TL = torch.randn(16, 5)
        gathered_k_shared_TR = torch.randn(16, 2)
        gathered_kv_THP = torch.randn(16, 4, 7)
        wkv_b = _MockProjection(gathered_kv_THP.flatten(-2))
        expected_k_THK = torch.cat(
            (
                gathered_kv_THP[..., :4],
                gathered_k_shared_TR.unsqueeze(1).expand(-1, 4, -1),
            ),
            dim=-1,
        )
        expected_v_THV = gathered_kv_THP[..., 4:]
        out_THV = torch.randn(8, 4, 3)
        attention_mask = object()

        kernel = AllGatherKVMLAFlexAttention(
            AllGatherKVMLAFlexAttention.Config(**_mla_config_kwargs())
        )
        kernel.wkv_b = wkv_b
        calls = []

        def record(x, group, *, src, dst, backward_options=None):
            calls.append((x, group, src, dst, backward_options))
            if x is kv_c_normed_TL:
                return gathered_kv_c_normed_TL
            if x is k_shared_TR:
                return gathered_k_shared_TR
            raise AssertionError("Only compressed MLA inputs should be communicated")

        with _in_mesh(2), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(
            FlexInnerAttention,
            "forward",
            autospec=True,
            return_value=out_THV,
        ) as flex_forward:
            result = kernel.forward(
                q_THK,
                kv_c_normed_TL,
                k_shared_TR,
                attention_metadata=attention_mask,
            )

        self.assertIs(result, out_THV)
        wkv_b.call.assert_called_once_with(gathered_kv_c_normed_TL)
        self.assertEqual(2, len(calls))
        self.assertIs(calls[0][0], kv_c_normed_TL)
        self.assertIs(calls[1][0], k_shared_TR)
        for _, group, src, dst, backward_options in calls:
            self.assertEqual(2, group.size())
            self.assertEqual(spmd.S(0), src)
            self.assertEqual(spmd.R, dst)
            self.assertEqual({"op_dtype": torch.float32}, backward_options)
        flex_forward.assert_called_once()
        args = flex_forward.call_args.args
        self.assertIs(args[0], kernel.inner_attention)
        self.assertIs(args[1], q_THK)
        torch.testing.assert_close(args[2], expected_k_THK)
        torch.testing.assert_close(args[3], expected_v_THV)
        self.assertIs(
            flex_forward.call_args.kwargs["attention_metadata"], attention_mask
        )

    def test_ulysses_keeps_shared_key_headless_during_communication(self):
        q_THK = torch.randn(8, 4, 6)
        kv_c_normed_TL = torch.randn(8, 5)
        kv_THP = torch.randn(8, 4, 7)
        k_shared_TR = torch.randn(8, 2)
        wkv_b = _MockProjection(kv_THP.flatten(-2))
        expected_k_THK = torch.cat(
            (kv_THP[..., :4], k_shared_TR.unsqueeze(1).expand(-1, 4, -1)), dim=-1
        )
        expected_v_THV = kv_THP[..., 4:]
        out_THV = torch.randn(8, 4, 3)
        calls = []

        def record(x, group, *, src, dst, backward_options=None, async_op=False):
            calls.append((x, group, src, dst, backward_options, async_op))
            return x

        config = UlyssesMLAAttention.Config(**_mla_config_kwargs())
        self.assertIsInstance(config, InnerAttention.Config)
        kernel = UlyssesMLAAttention(config)
        self.assertIs(
            kernel.prepare_cp_metadata, UlyssesInnerAttention.prepare_cp_metadata
        )
        kernel.wkv_b = wkv_b
        with _in_mesh(2), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(
            FlexInnerAttention,
            "forward",
            autospec=True,
            return_value=out_THV,
        ) as inner_forward:
            result = kernel.forward(
                q_THK,
                kv_c_normed_TL,
                k_shared_TR,
                attention_metadata=None,
            )

        self.assertIs(result, out_THV)
        wkv_b.call.assert_called_once_with(kv_c_normed_TL)
        self.assertEqual(3, len(calls))
        self.assertIs(calls[0][0], k_shared_TR)
        self.assertEqual((spmd.S(0), spmd.R), calls[0][2:4])
        self.assertEqual({"op_dtype": k_shared_TR.dtype}, calls[0][4])
        self.assertEqual((8, 4, 13), calls[1][0].shape)
        self.assertEqual((spmd.S(0), spmd.S(1)), calls[1][2:4])
        self.assertEqual((spmd.S(1), spmd.S(0)), calls[2][2:4])
        self.assertTrue(all(call[5] for call in calls))
        inner_forward.assert_called_once()
        args = inner_forward.call_args.args
        self.assertIs(args[0], kernel.inner_attention)
        torch.testing.assert_close(args[1], q_THK)
        torch.testing.assert_close(args[2], expected_k_THK)
        torch.testing.assert_close(args[3], expected_v_THV)


class TestUlyssesVarlen(unittest.TestCase):
    def test_is_still_a_varlen_kernel(self):
        config = UlyssesVarlenInnerAttention.Config()
        self.assertIsInstance(config, UlyssesInnerAttention.Config)
        self.assertIsInstance(config, VarlenInnerAttention.Config)

    def test_dispatches_to_varlen_inner_attention(self):
        q, k, v = (torch.randn(8, 4, 16) for _ in range(3))
        mask = object()
        kernel = UlyssesVarlenInnerAttention(UlyssesVarlenInnerAttention.Config())

        with _in_mesh(2), mock.patch.object(
            spmd,
            "redistribute",
            side_effect=lambda x, *_args, **_kwargs: x,
        ), mock.patch.object(
            VarlenInnerAttention,
            "forward",
            autospec=True,
            return_value=q,
        ) as inner_forward:
            result = kernel.forward(q, k, v, attention_metadata=mask)

        inner_forward.assert_called_once_with(kernel, q, k, v, attention_metadata=mask)
        self.assertIs(result, q)


if __name__ == "__main__":
    unittest.main()
