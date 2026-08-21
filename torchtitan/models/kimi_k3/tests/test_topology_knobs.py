# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Topology knobs come from config, and the env fallback is visible (finding 32).

Five knobs decided the pipeline topology from environment variables. The hazard was
never that an env var is ugly: a launcher exporting them non-uniformly gives ranks
DIFFERENT topologies, which hangs in a collective with nothing naming the cause, and
a run is not reproducible from its config or checkpoint.

What has to hold, and what these pin:

* the config field is the source of truth;
* the retired env name still overrides it -- a dozen recorded repro commands set
  them, and silently ignoring those would make every one of those documents wrong;
* reading a knob before any registration is WARNED, not silent, because that path
  does not honour config at all;
* re-registering with a different resolution keeps the first and says so, so the
  answer cannot depend on which entry point ran first.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest import mock

from torchtitan.models.kimi_k3.knobs import (
    register_topology,
    reset_topology_for_testing,
    resolve_knob,
    topology,
    TopologyKnobs,
)


@dataclass
class _TextCfg:
    attn_res_cache: bool = False


@dataclass
class _Cfg:
    kimi_config: _TextCfg
    vit_dep: bool = False
    vit_dep_stages: int = 1
    vit_prefetch: int = 0
    vit_tp_heads: bool = True


class TestResolveKnob(unittest.TestCase):
    def setUp(self):
        reset_topology_for_testing()

    def test_config_is_the_source_of_truth(self):
        cfg = _Cfg(kimi_config=_TextCfg(), vit_dep=True, vit_dep_stages=2)
        with mock.patch.dict("os.environ", {}, clear=True):
            t = register_topology(cfg)
        self.assertTrue(t.vit_dep)
        self.assertEqual(t.vit_dep_stages, 2)

    def test_env_still_overrides_a_config_field(self):
        cfg = _Cfg(kimi_config=_TextCfg(), vit_dep=False)
        with mock.patch.dict("os.environ", {"KIMI_VIT_DEP": "1"}, clear=True):
            t = register_topology(cfg)
        self.assertTrue(t.vit_dep, "a recorded repro command must keep working")

    def test_zero_is_off_for_booleans(self):
        """The historical convention exactly: '0' off, anything else on."""
        cfg = _Cfg(kimi_config=_TextCfg(), vit_tp_heads=True)
        with mock.patch.dict("os.environ", {"KIMI_VIT_TP_HEADS": "0"}, clear=True):
            t = register_topology(cfg)
        self.assertFalse(t.vit_tp_heads)

    def test_int_knobs_are_typed_from_the_default(self):
        cfg = _Cfg(kimi_config=_TextCfg())
        with mock.patch.dict("os.environ", {"KIMI_VIT_DEP_STAGES": "3"}, clear=True):
            t = register_topology(cfg)
        self.assertEqual(t.vit_dep_stages, 3)
        self.assertIsInstance(t.vit_dep_stages, int)

    def test_the_adapter_gate_comes_off_the_text_config(self):
        """It gates the PP adapter for text flavors too, so it does not live on the
        multimodal config; the multimodal one reaches it through kimi_config."""
        cfg = _Cfg(kimi_config=_TextCfg(attn_res_cache=True))
        with mock.patch.dict("os.environ", {}, clear=True):
            t = register_topology(cfg)
        self.assertTrue(t.attn_res_cache)

    def test_a_config_without_the_fields_still_runs(self):
        """Flavors built before these fields existed must not crash."""

        @dataclass
        class _Old:
            pass

        with mock.patch.dict("os.environ", {"KIMI_VIT_DEP": "1"}, clear=True):
            t = register_topology(_Old())
        self.assertTrue(t.vit_dep)
        self.assertEqual(t.vit_dep_stages, 1)

    def test_first_registration_wins_and_a_disagreement_is_reported(self):
        a = _Cfg(kimi_config=_TextCfg(), vit_dep=True)
        b = _Cfg(kimi_config=_TextCfg(), vit_dep=False)
        with mock.patch.dict("os.environ", {}, clear=True):
            first = register_topology(a)
            with self.assertLogs(level="WARNING") as logs:
                second = register_topology(b)
        self.assertTrue(first.vit_dep)
        self.assertTrue(second.vit_dep, "first call must win")
        self.assertTrue(
            any("re-registered" in line for line in logs.output),
            "a disagreement between the two entry points must be reported",
        )

    def test_reading_before_registration_warns(self):
        with mock.patch.dict("os.environ", {"KIMI_VIT_DEP": "1"}, clear=True):
            with self.assertLogs(level="WARNING") as logs:
                t = topology()
        self.assertTrue(t.vit_dep, "the env fallback still answers")
        self.assertTrue(
            any("before register_topology" in line for line in logs.output),
            "this path does not honour config and must say so",
        )

    def test_defaults_match_the_historical_env_defaults(self):
        """A behaviour change hidden in a default would be invisible in review."""
        d = TopologyKnobs()
        self.assertFalse(d.vit_dep)
        self.assertEqual(d.vit_dep_stages, 1)
        self.assertEqual(d.vit_prefetch, 0)
        self.assertTrue(d.vit_tp_heads)
        self.assertFalse(d.attn_res_cache)

    def test_resolve_knob_warns_once_per_variable(self):
        cfg = _Cfg(kimi_config=_TextCfg())
        with mock.patch.dict("os.environ", {"KIMI_VIT_DEP": "1"}, clear=True):
            with self.assertLogs(level="WARNING"):
                resolve_knob(cfg, "vit_dep", "KIMI_VIT_DEP")
            # Second read must not warn again; assertLogs fails when nothing is logged,
            # so the absence of a warning is what makes this pass.
            with self.assertRaises(AssertionError):
                with self.assertLogs(level="WARNING"):
                    resolve_knob(cfg, "vit_dep", "KIMI_VIT_DEP")


if __name__ == "__main__":
    unittest.main()
