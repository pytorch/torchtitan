# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

from torchtitan.components.optimizer import EMA


class TestEMADynamicDecay(unittest.TestCase):
    """CPU-only: verifies the dynamic (half_life_fraction-based) decay
    schedule -- in particular that update_every_n_steps != 1 uses the EMA
    firing count (not raw elapsed steps) as the schedule's age, since the
    half-life formula's math is defined in terms of applications of decay.
    """

    def test_default_n1_matches_closed_form(self):
        model = nn.Linear(4, 4)
        ema = EMA.Config().build(model_parts=[model])
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        for step in range(1, 6):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            beta = 2.0 ** (-1.0 / (0.05 * step))
            expected = expected * beta + model.weight.detach() * (1 - beta)
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_update_every_n_steps_uses_firing_count_not_raw_steps(self):
        """Regression test: the schedule's age must be the number of EMA
        firings, not the raw step count, or it ages far faster than
        intended whenever update_every_n_steps > 1."""
        model = nn.Linear(4, 4)
        ema = EMA.Config(update_every_n_steps=2).build(model_parts=[model])
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        fire_count = 0
        for step in range(1, 11):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            if step % 2 != 0:
                continue
            fire_count += 1
            beta = 2.0 ** (-1.0 / (0.05 * fire_count))
            expected = expected * beta + model.weight.detach() * (1 - beta)
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        self.assertEqual(fire_count, 5)

    def test_step_bias_renumbers_aging(self):
        """step_bias lets a deliberate Trainer.step reset keep the EMA aging
        as if training had continued uninterrupted."""
        model = nn.Linear(4, 4)
        ema = EMA.Config(step_bias=6).build(model_parts=[model])
        start = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        with torch.no_grad():
            model.weight.fill_(7.0)
        ema.step(1)  # current_step=1, step_bias=6 -> num_updates = 7
        beta7 = 2.0 ** (-1.0 / (0.05 * 7))
        expected = start * beta7 + model.weight.detach() * (1 - beta7)
        actual = ema.optimizers[0].state[model.weight]["ema_params"]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_fixed_decay_ignores_firing_count(self):
        """A fixed `decay` bypasses the half-life schedule entirely -- same
        value regardless of how many times it's fired."""
        model = nn.Linear(4, 4)
        ema = EMA.Config(decay=0.9).build(model_parts=[model])
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        for step in range(1, 4):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            expected = expected * 0.9 + model.weight.detach() * 0.1
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_start_step_does_not_double_count_first_firing(self):
        """Regression test: with start_step > 0 the firing count must advance
        1, 2, 3, ... rather than getting stuck at 1 for the first two firings.
        start_step is the last step before tracking begins, so step 10 with
        start_step=10 does not fire and step 11 is firing number 1."""
        model = nn.Linear(4, 4)
        ema = EMA.Config(start_step=10).build(model_parts=[model])
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()

        with torch.no_grad():
            model.weight.fill_(5.0)
        ema.step(10)  # elapsed == 0 -> no firing yet
        torch.testing.assert_close(
            ema.optimizers[0].state[model.weight]["ema_params"],
            expected,
            atol=0,
            rtol=0,
        )

        for i, step in enumerate((11, 12, 13), start=1):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            beta = 2.0 ** (-1.0 / (0.05 * i))
            expected = expected * beta + model.weight.detach() * (1 - beta)
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_decay_schedule_survives_checkpoint_resume(self):
        """Regression test: the decay schedule must be derived from
        current_step, not a counter held on the instance. Only ema_params are
        checkpointed, so a stored firing count would restart at 0 on resume
        and the next update's decay would collapse to ~0, overwriting the
        restored EMA with the current weights."""
        import torch.distributed.checkpoint as dcp

        model = nn.Linear(2, 2)
        ema = EMA.Config().build(model_parts=[model])
        with torch.no_grad():
            model.weight.fill_(1.0)
        for step in range(1, 2001):
            ema.step(step)
        saved = ema.optimizers[0].state[model.weight]["ema_params"].clone()

        ckpt_dir = tempfile.mkdtemp()
        try:
            dcp.save({"ema": ema}, checkpoint_id=ckpt_dir)

            model2 = nn.Linear(2, 2)
            ema2 = EMA.Config().build(model_parts=[model2])
            dcp.load({"ema": ema2}, checkpoint_id=ckpt_dir)
            torch.testing.assert_close(
                ema2.optimizers[0].state[model2.weight]["ema_params"],
                saved,
                atol=1e-6,
                rtol=0,
            )

            # One post-resume step against a wildly different weight must
            # barely move the EMA, since step 2001 is deep in the schedule.
            with torch.no_grad():
                model2.weight.fill_(99.0)
            ema2.step(2001)
            after = ema2.optimizers[0].state[model2.weight]["ema_params"]
            beta = 2.0 ** (-1.0 / (0.05 * 2001))
            expected = saved * beta + model2.weight.detach() * (1 - beta)
            torch.testing.assert_close(after, expected, atol=1e-5, rtol=0)
            self.assertLess(after.max().item(), 2.0)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_step_bias_with_update_every_n_steps_not_a_multiple(self):
        """Regression test: step_bias must shift the firing count even when
        it isn't an exact multiple of update_every_n_steps -- previously the
        two were combined via floor division in step-space, silently
        discarding step_bias's remainder."""
        model_a = nn.Linear(4, 4)
        model_b = nn.Linear(4, 4)
        ema_a = EMA.Config(update_every_n_steps=2, step_bias=0).build(
            model_parts=[model_a]
        )
        ema_b = EMA.Config(update_every_n_steps=2, step_bias=1).build(
            model_parts=[model_b]
        )
        for step in range(1, 9):
            with torch.no_grad():
                model_a.weight.fill_(float(step))
                model_b.weight.fill_(float(step))
            ema_a.step(step)
            ema_b.step(step)
        val_a = ema_a.optimizers[0].state[model_a.weight]["ema_params"]
        val_b = ema_b.optimizers[0].state[model_b.weight]["ema_params"]
        self.assertFalse(torch.equal(val_a, val_b))


class _ModelWithExpertBias(nn.Module):
    """Toy stand-in for a module with a non-gradient-updated buffer (e.g.
    MoE's expert_bias_E, updated by a load-balancing heuristic)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("expert_bias_E", torch.zeros(4))


class _ModelWithoutExpertBias(nn.Module):
    """PP-stage stand-in with no buffer matching buffer_patterns."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _ModelWithNonFloatBuffers(nn.Module):
    """A float buffer alongside integer and boolean ones, all matched by a
    single pattern."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("bias_buf", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("count_buf", torch.zeros(4, dtype=torch.int64))
        self.register_buffer("flag_buf", torch.zeros(4, dtype=torch.bool))


class TestEMABufferSupport(unittest.TestCase):
    """CPU-only: verifies buffer_patterns tracks matching buffers alongside
    parameters, folded into the same "ema" checkpoint key."""

    def test_matched_buffer_converges_like_a_parameter(self):
        model = _ModelWithExpertBias()
        ema = EMA.Config(buffer_patterns=["expert_bias_E"]).build(model_parts=[model])
        expected = (
            ema._buffer_optimizers[0].state[model.expert_bias_E]["ema_params"].clone()
        )
        for step in range(1, 6):
            with torch.no_grad():
                model.expert_bias_E.fill_(float(step))
            ema.step(step)
            beta = 2.0 ** (-1.0 / (0.05 * step))
            expected = expected * beta + model.expert_bias_E.detach() * (1 - beta)
            actual = ema._buffer_optimizers[0].state[model.expert_bias_E]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_default_empty_patterns_leaves_buffers_untracked(self):
        """Regression test: buffer_patterns defaults to [], so existing
        configs see zero behavior change -- no buffer_optimizers built."""
        model = _ModelWithExpertBias()
        ema = EMA.Config().build(model_parts=[model])
        self.assertEqual(ema._buffer_optimizers, [])
        ema.step(1)  # must not error despite the untracked buffer existing

    def test_checkpoint_round_trip_folds_buffer_into_ema_key(self):
        import torch.distributed.checkpoint as dcp

        model = _ModelWithExpertBias()
        ema = EMA.Config(buffer_patterns=["expert_bias_E"]).build(model_parts=[model])
        with torch.no_grad():
            model.expert_bias_E.fill_(3.0)
        ema.step(1)
        saved_buffer_ema = (
            ema._buffer_optimizers[0].state[model.expert_bias_E]["ema_params"].clone()
        )

        ckpt_dir = tempfile.mkdtemp()
        try:
            state_dict = {"ema": ema}
            dcp.save(state_dict, checkpoint_id=ckpt_dir)
            # No separate "ema_buffer" key -- the buffer's FQN is folded into
            # the same flat state dict alongside parameter FQNs.
            self.assertIn("state.expert_bias_E.ema_params", ema.state_dict())

            model2 = _ModelWithExpertBias()
            ema2 = EMA.Config(buffer_patterns=["expert_bias_E"]).build(
                model_parts=[model2]
            )
            dcp.load({"ema": ema2}, checkpoint_id=ckpt_dir)

            actual = ema2._buffer_optimizers[0].state[model2.expert_bias_E][
                "ema_params"
            ]
            torch.testing.assert_close(actual, saved_buffer_ema, atol=1e-6, rtol=0)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_multi_part_with_one_buffer_free_part_is_safe(self):
        model_a = _ModelWithExpertBias()
        model_b = _ModelWithoutExpertBias()
        ema = EMA.Config(buffer_patterns=["expert_bias_E"]).build(
            model_parts=[model_a, model_b]
        )
        self.assertEqual(len(ema._buffer_optimizers), 2)
        self.assertEqual(len(ema._buffer_optimizers[1].state), 0)  # model_b: empty

        expected = (
            ema._buffer_optimizers[0].state[model_a.expert_bias_E]["ema_params"].clone()
        )
        for step in range(1, 6):
            with torch.no_grad():
                model_a.expert_bias_E.fill_(float(step))
            ema.step(step)  # must not raise for either part
            beta = 2.0 ** (-1.0 / (0.05 * step))
            expected = expected * beta + model_a.expert_bias_E.detach() * (1 - beta)
            actual = ema._buffer_optimizers[0].state[model_a.expert_bias_E][
                "ema_params"
            ]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_non_float_buffer_is_rejected(self):
        """An integer or boolean average has to be rounded back into the
        buffer's dtype, which freezes it once the per-step increment drops
        below one (and latches a bool to True on the first firing), so such a
        buffer is refused at build time rather than silently mistracked."""
        model = _ModelWithNonFloatBuffers()
        for pattern, dtype in (("count_buf$", "int64"), ("flag_buf$", "bool")):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError) as caught:
                    EMA.Config(buffer_patterns=[pattern]).build(model_parts=[model])
                self.assertIn(dtype, str(caught.exception))

    def test_float_buffer_alongside_non_float_is_still_rejected(self):
        """One pattern matching a mix must fail rather than tracking only the
        float buffers."""
        model = _ModelWithNonFloatBuffers()
        with self.assertRaises(ValueError):
            EMA.Config(buffer_patterns=["_buf$"]).build(model_parts=[model])

    def test_update_does_not_build_an_autograd_graph(self):
        """The update must run under no_grad. Without it, the in-place lerp on
        ema_params chains a new grad_fn every firing, so the graph grows for
        the whole run and the checkpointed tensor is a non-leaf."""
        model = _ModelWithExpertBias()
        ema = EMA.Config(buffer_patterns=["expert_bias_E$"]).build(model_parts=[model])
        self.assertTrue(torch.is_grad_enabled())  # as the trainer calls it
        for step in range(1, 6):
            ema.step(step)
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                stored = param_state["ema_params"]
                self.assertFalse(stored.requires_grad)
                self.assertIsNone(stored.grad_fn)


class TestEMATrackedSetIsFixed(unittest.TestCase):
    """EMA state is keyed by tensor identity and the tracked set is fixed when
    the container is built. Changing it afterwards has to fail clearly, not
    with a bare KeyError from the defaultdict behind the state.
    """

    def test_unfreezing_a_parameter_afterwards_fails_clearly(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model[1].weight.requires_grad_(False)
        ema = EMA.Config().build(model_parts=[model])
        model[1].weight.requires_grad_(True)
        with self.assertRaises(RuntimeError) as caught:
            ema.step(1)
        self.assertIn("EMA has no state", str(caught.exception))

    def test_replacing_a_parameter_afterwards_fails_clearly(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        ema = EMA.Config().build(model_parts=[model])
        model[0].weight = nn.Parameter(torch.zeros(4, 4))
        with self.assertRaises(RuntimeError) as caught:
            ema.step(1)
        self.assertIn("EMA has no state", str(caught.exception))

    def test_replacing_a_tracked_buffer_afterwards_fails_clearly(self):
        model = _ModelWithExpertBias()
        ema = EMA.Config(buffer_patterns=[r"expert_bias_E$"]).build(model_parts=[model])
        model.expert_bias_E = torch.ones_like(model.expert_bias_E)
        with self.assertRaises(RuntimeError) as caught:
            ema.step(1)
        self.assertIn("EMA has no state", str(caught.exception))

    def test_freezing_a_parameter_afterwards_is_allowed(self):
        """The reverse is fine: a frozen parameter just stops being averaged."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        ema = EMA.Config().build(model_parts=[model])
        model[0].weight.requires_grad_(False)
        ema.step(1)  # must not raise

    def test_the_failed_lookup_does_not_pollute_state(self):
        """state is a defaultdict, so a careless lookup would insert an empty
        entry and corrupt the next checkpoint."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model[1].weight.requires_grad_(False)
        ema = EMA.Config().build(model_parts=[model])
        before = len(ema.state_dict())
        model[1].weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            ema.step(1)
        self.assertEqual(len(ema.state_dict()), before)


class TestEMAWarnings(unittest.TestCase):
    """Two ways the EMA can silently do nothing useful."""

    def test_cold_start_warns_that_history_is_discarded(self):
        """load_state_dict({}) is how a resume reseeds the EMA from the loaded
        weights. Correct the first time EMA is switched on, but if
        exclude_from_loading keeps listing "ema" it silently happens on every
        later resume, throwing away all EMA history."""
        model = _ModelWithExpertBias()
        ema = EMA.Config().build(model_parts=[model])
        with self.assertLogs(
            "torchtitan.components.optimizer.ema", level="WARNING"
        ) as logs:
            ema.load_state_dict({})
        joined = "\n".join(logs.output)
        self.assertIn("discards all EMA history", joined)
        self.assertIn("exclude_from_loading", joined)

    def test_half_life_fraction_with_fixed_decay_warns(self):
        """A fixed decay replaces the half-life schedule, so setting both
        means half_life_fraction silently does nothing."""
        with self.assertLogs(
            "torchtitan.components.optimizer.ema", level="WARNING"
        ) as logs:
            EMA.Config(decay=0.9, half_life_fraction=0.42)
        joined = "\n".join(logs.output)
        self.assertIn("half_life_fraction", joined)
        self.assertIn("ignored", joined)

    def test_fixed_decay_alone_does_not_warn(self):
        with self.assertNoLogs("torchtitan.components.optimizer.ema", level="WARNING"):
            EMA.Config(decay=0.9)

    def test_half_life_fraction_alone_does_not_warn(self):
        with self.assertNoLogs("torchtitan.components.optimizer.ema", level="WARNING"):
            EMA.Config(half_life_fraction=0.42)

    def test_low_precision_tracking_warns(self):
        """bfloat16 has 8 mantissa bits, so the per-firing increment rounds
        away and the EMA never moves."""
        model = _ModelWithExpertBias().to(torch.bfloat16)
        with self.assertLogs(
            "torchtitan.components.optimizer.ema", level="WARNING"
        ) as logs:
            EMA.Config().build(model_parts=[model])
        self.assertIn("torch.bfloat16", "\n".join(logs.output))

    def test_float32_tracking_does_not_warn(self):
        model = _ModelWithExpertBias()
        with self.assertNoLogs("torchtitan.components.optimizer.ema", level="WARNING"):
            EMA.Config().build(model_parts=[model])

    def test_bf16_increment_rounds_away_while_float32_tracks(self):
        """Pins the reason the warning exists. At a late firing the decay is
        near 1, so the increment is small: float32 still moves, bfloat16
        rounds the whole increment away and stops tracking."""
        moved = {}
        for dtype in (torch.float32, torch.bfloat16):
            model = nn.Linear(8, 8, bias=False).to(dtype)
            with torch.no_grad():
                model.weight.fill_(1.0)
            ema = EMA.Config().build(model_parts=[model])
            with torch.no_grad():
                model.weight.fill_(2.0)
            # num_updates is derived from the step, so this is firing 10000:
            # decay = 2 ** (-1 / (0.05 * 10000)), i.e. an increment of ~1.4e-3
            ema.step(10000)
            stored = ema.optimizers[0].state[model.weight]["ema_params"]
            moved[dtype] = stored.flatten()[0].float().item() - 1.0
        self.assertGreater(moved[torch.float32], 0.0)
        self.assertEqual(moved[torch.bfloat16], 0.0)


class TestEMAConfigValidation(unittest.TestCase):
    """Config values that would otherwise corrupt the EMA silently."""

    def test_update_every_n_steps_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            EMA.Config(update_every_n_steps=0)

    def test_non_positive_half_life_fraction_is_rejected(self):
        for bad in (0.0, -0.05):
            with self.subTest(half_life_fraction=bad), self.assertRaises(ValueError):
                EMA.Config(half_life_fraction=bad)

    def test_decay_outside_unit_interval_is_rejected(self):
        # decay=1.0 is rejected because it never updates the EMA at all.
        for bad in (-1.0, 1.0, 2.0):
            with self.subTest(decay=bad), self.assertRaises(ValueError):
                EMA.Config(decay=bad)

    def test_negative_step_bias_is_rejected(self):
        # step_bias is added to the firing count; a non-positive count gives
        # decay > 1 (divergence) or a ZeroDivisionError.
        for bad in (-1, -2):
            with self.subTest(step_bias=bad), self.assertRaises(ValueError):
                EMA.Config(step_bias=bad)

    def test_non_finite_values_are_rejected(self):
        # nan slips past a `<= 0` guard and silently makes the whole EMA nan.
        nan, inf = float("nan"), float("inf")
        for kwargs in (
            {"half_life_fraction": nan},
            {"half_life_fraction": inf},
            {"decay": nan},
            {"decay": inf},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                EMA.Config(**kwargs)

    def test_valid_values_are_accepted(self):
        EMA.Config()
        EMA.Config(decay=0.0)
        EMA.Config(decay=0.999)
        EMA.Config(update_every_n_steps=4, half_life_fraction=0.1)
        # negative start_step only shifts the schedule, so it stays allowed
        EMA.Config(start_step=-100)


if __name__ == "__main__":
    unittest.main()
