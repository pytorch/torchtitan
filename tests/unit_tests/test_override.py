# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field

from torchtitan.config import (
    apply_overrides,
    clear_overrides,
    Configurable,
    derive,
    override,
    OverrideConfig,
)
from torchtitan.config.override import _REGISTRY, Override

# Overrides registered in this module have ``origin_module == __name__``; tests
# pass ``imports=[__name__]`` so the provenance filter activates them.


class ComponentA(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 64

    def __init__(self, config: Config):
        self.config = config


class ComponentB(Configurable):
    """Replacement for ComponentA."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 64
        extra: int = 128

    def __init__(self, config: Config):
        self.config = config


class ParentComponent(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        child: ComponentA.Config = field(default_factory=ComponentA.Config)
        children: list[ComponentA.Config] = field(
            default_factory=lambda: [
                ComponentA.Config(dim=32),
                ComponentA.Config(dim=48),
            ]
        )

    def __init__(self, config: Config):
        self.config = config


class Root(Configurable):
    """Wrapper so ParentComponent appears as a nested node (for nesting tests)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        block: ParentComponent.Config = field(default_factory=ParentComponent.Config)

    def __init__(self, config: Config):
        self.config = config


class TwoBlocks(Configurable):
    """Two independent ParentComponent subtrees (for disjoint-subtree tests)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        a: ParentComponent.Config = field(default_factory=ParentComponent.Config)
        b: ParentComponent.Config = field(default_factory=ParentComponent.Config)

    def __init__(self, config: Config):
        self.config = config


class DerivedComponent(ComponentA):
    """A replacement that subclasses the target Config and adds a field.

    Mirrors the common override shape (e.g. ``TritonRoPE.Config(RoPE.Config)``):
    it inherits the target's fields, so any field later added to the target is
    inherited here too and ``derive`` carries it automatically.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ComponentA.Config):  # inherits ``dim``
        block_size: int = 128

    def __init__(self, config: Config):
        self.config = config


def _to_b(cfg: ComponentA.Config) -> ComponentB.Config:
    return ComponentB.Config(dim=cfg.dim)


class TestOverride(unittest.TestCase):
    def setUp(self):
        clear_overrides()

    def tearDown(self):
        clear_overrides()

    def _imports(self) -> OverrideConfig:
        """OverrideConfig that activates overrides defined in this module."""
        return OverrideConfig(imports=[__name__])

    def test_register_and_apply(self):
        @override("test_swap", target=ComponentA.Config, description="test swap")
        def swap_a_to_b(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=256)

        self.assertIn("test_swap", _REGISTRY)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(self._imports(), parent_cfg)

        # No `where` -> child and both list items are replaced.
        self.assertEqual(len(replacements), 3)
        self.assertIsInstance(parent_cfg.child, ComponentB.Config)
        self.assertEqual(parent_cfg.child.extra, 256)
        for child_cfg in parent_cfg.children:
            self.assertIsInstance(child_cfg, ComponentB.Config)

    def test_fqns_glob(self):
        @override("by_glob", target=ComponentA.Config, fqns=["children.*"])
        def by_glob(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(self._imports(), parent_cfg)

        self.assertEqual(len(replacements), 2)
        self.assertIsInstance(parent_cfg.child, ComponentA.Config)  # "child", skipped
        for child_cfg in parent_cfg.children:
            self.assertIsInstance(child_cfg, ComponentB.Config)

    def test_fqns_glob_list(self):
        @override("by_globs", target=ComponentA.Config, fqns=["child", "nomatch.*"])
        def by_globs(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(self._imports(), parent_cfg)

        self.assertEqual(len(replacements), 1)
        self.assertIsInstance(parent_cfg.child, ComponentB.Config)
        for child_cfg in parent_cfg.children:
            self.assertIsInstance(child_cfg, ComponentA.Config)

    def test_same_class_disjoint_fqns_ok(self):
        # Two overrides of the SAME class, but disjoint FQNs -> no conflict.
        @override("on_child", target=ComponentA.Config, fqns=["child"])
        def on_child(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=1)

        @override("on_children", target=ComponentA.Config, fqns=["children.*"])
        def on_children(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=2)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(self._imports(), parent_cfg)

        self.assertEqual(len(replacements), 3)
        self.assertEqual(parent_cfg.child.extra, 1)
        for child_cfg in parent_cfg.children:
            self.assertEqual(child_cfg.extra, 2)

    def test_exact_target_skips_subclasses(self):
        @override("exact_a", target=ComponentA.Config, exact=True)
        def exact_a(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=10)

        parent_cfg = ParentComponent.Config(
            child=DerivedComponent.Config(dim=99),
            children=[
                ComponentA.Config(dim=32),
                DerivedComponent.Config(dim=48),
            ],
        )
        replacements = apply_overrides(self._imports(), parent_cfg)

        self.assertEqual(len(replacements), 1)
        self.assertIs(type(parent_cfg.child), DerivedComponent.Config)
        self.assertIsInstance(parent_cfg.children[0], ComponentB.Config)
        self.assertEqual(parent_cfg.children[0].extra, 10)
        self.assertIs(type(parent_cfg.children[1]), DerivedComponent.Config)

    def test_exact_target_does_not_conflict_with_subclass_override(self):
        @override("exact_a", target=ComponentA.Config, exact=True)
        def exact_a(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=10)

        @override("derived", target=DerivedComponent.Config)
        def derived(cfg: DerivedComponent.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=20)

        parent_cfg = ParentComponent.Config(
            child=DerivedComponent.Config(dim=99),
            children=[
                ComponentA.Config(dim=32),
                DerivedComponent.Config(dim=48),
            ],
        )
        replacements = apply_overrides(self._imports(), parent_cfg)

        self.assertEqual(len(replacements), 3)
        self.assertEqual(parent_cfg.child.extra, 20)
        self.assertEqual(parent_cfg.children[0].extra, 10)
        self.assertEqual(parent_cfg.children[1].extra, 20)

    def test_non_configurable_target_raises(self):
        # `target` must be a Configurable.Config subclass; a plain class (e.g.
        # ModelSpec) or non-type is rejected at registration.
        with self.assertRaisesRegex(TypeError, "Configurable.Config subclass"):

            @override("bad_target", target=dict)
            def bad(cfg):
                return cfg

    def test_conflict_same_node_raises(self):
        # Two overrides claim the same nodes (both target ComponentA, no `where`).
        @override("swap1", target=ComponentA.Config)
        def swap1(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        @override("swap2", target=ComponentA.Config)
        def swap2(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        parent_cfg = ParentComponent.Config()
        with self.assertRaisesRegex(ValueError, "both claim node"):
            apply_overrides(self._imports(), parent_cfg)

    def test_parent_child_disjoint_subtrees_ok(self):
        # A targets the parent class on subtree "a"; B targets the child class
        # on subtree "b". Disjoint nodes -> both apply, no conflict.
        @override("on_parent", target=ParentComponent.Config, fqns=["a"])
        def on_parent(cfg: ParentComponent.Config) -> ComponentB.Config:
            return ComponentB.Config()

        @override("on_child", target=ComponentA.Config, fqns=["b.*"])
        def on_child(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        root_cfg = TwoBlocks.Config()
        replacements = apply_overrides(self._imports(), root_cfg)

        self.assertEqual(len(replacements), 4)  # "a" + b.child + b.children[0,1]
        self.assertIsInstance(root_cfg.a, ComponentB.Config)  # whole parent
        self.assertIsInstance(root_cfg.b, ParentComponent.Config)  # untouched parent
        self.assertIsInstance(root_cfg.b.child, ComponentB.Config)
        for child_cfg in root_cfg.b.children:
            self.assertIsInstance(child_cfg, ComponentB.Config)

    def test_conflict_nested_node_raises(self):
        # One override claims an ancestor node of another's node.
        @override("on_parent", target=ParentComponent.Config)
        def on_parent(cfg: ParentComponent.Config) -> ComponentB.Config:
            return ComponentB.Config()  # never built; error is raised first

        @override("on_leaf", target=ComponentA.Config)
        def on_leaf(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        root_cfg = Root.Config()
        with self.assertRaisesRegex(ValueError, "ancestor"):
            apply_overrides(self._imports(), root_cfg)

    def test_provenance_only_listed_modules_apply(self):
        @override("local", target=ComponentA.Config)
        def local(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=256)

        # An override registered by a module the user did NOT list. Insert it
        # directly with a foreign origin (the @override decorator would capture
        # this module's name).
        def foreign_factory(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=999)

        _REGISTRY["foreign"] = Override(
            name="foreign",
            target_cls=ComponentA.Config,
            factory=foreign_factory,
            fqns=None,
            description="",
            origin_module="vendor.unlisted",
        )

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(self._imports(), parent_cfg)

        # Only "local" applied; "foreign" filtered out by provenance (so no
        # same-node conflict was raised despite both targeting ComponentA).
        self.assertEqual(parent_cfg.child.extra, 256)
        self.assertTrue(all("foreign" not in line for line in replacements))
        self.assertTrue(all("local" in line for line in replacements))

    def test_duplicate_name_raises(self):
        @override("dup", target=ComponentA.Config)
        def first(cfg):
            return _to_b(cfg)

        with self.assertRaisesRegex(ValueError, "already registered"):

            @override("dup", target=ComponentA.Config)
            def second(cfg):
                return _to_b(cfg)

    def test_clear_overrides(self):
        @override("temp", target=ComponentA.Config)
        def temp(cfg):
            return _to_b(cfg)

        self.assertEqual(len(_REGISTRY), 1)
        clear_overrides()
        self.assertEqual(len(_REGISTRY), 0)

    def test_no_overrides_is_noop(self):
        parent_cfg = ParentComponent.Config(child=ComponentA.Config(dim=100))
        replacements = apply_overrides(OverrideConfig(), parent_cfg)
        self.assertEqual(len(replacements), 0)
        self.assertEqual(parent_cfg.child.dim, 100)

    def test_unlisted_imports_is_noop(self):
        # Override exists in the registry but the user lists no imports.
        @override("present", target=ComponentA.Config)
        def present(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(OverrideConfig(), parent_cfg)
        self.assertEqual(len(replacements), 0)
        self.assertIsInstance(parent_cfg.child, ComponentA.Config)

    def test_bad_module_import_raises(self):
        override_cfg = OverrideConfig(imports=["nonexistent.module.path"])
        parent_cfg = ParentComponent.Config()
        with self.assertRaises(ImportError):
            apply_overrides(override_cfg, parent_cfg)

    def test_logging_format(self):
        @override("fmt_test", target=ComponentA.Config, description="format test")
        def fmt_swap(cfg: ComponentA.Config) -> ComponentB.Config:
            return _to_b(cfg)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(self._imports(), parent_cfg)
        for line in replacements:
            self.assertIn("[Override]", line)
            self.assertIn("fmt_test", line)
            self.assertIn("->", line)


class TestOverrideKwargs(unittest.TestCase):
    """``override.imports`` entries may carry kwargs forwarded to the factory."""

    def setUp(self):
        clear_overrides()

    def tearDown(self):
        clear_overrides()

    def test_same_module_different_kwargs_per_actor(self):
        # The motivating case: two config trees share one override module but
        # pass different kwargs (e.g. RL trainer vs. generator capacity factor).
        @override("per_actor", target=ComponentA.Config, fqns=["child"])
        def per_actor(cfg: ComponentA.Config, *, extra: int) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=extra)

        trainer_cfg = ParentComponent.Config()
        generator_cfg = ParentComponent.Config()
        apply_overrides(OverrideConfig(imports=[(__name__, {"extra": 1})]), trainer_cfg)
        apply_overrides(
            OverrideConfig(imports=[(__name__, {"extra": 2})]), generator_cfg
        )

        self.assertEqual(trainer_cfg.child.extra, 1)
        self.assertEqual(generator_cfg.child.extra, 2)

    def test_bare_string_entry_calls_factory_without_kwargs(self):
        # A bare-string entry keeps the pre-kwargs contract: no kwargs passed.
        @override("no_kw", target=ComponentA.Config, fqns=["child"])
        def no_kw(cfg: ComponentA.Config, *, extra: int = 9) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=extra)

        parent_cfg = ParentComponent.Config()
        apply_overrides(OverrideConfig(imports=[__name__]), parent_cfg)
        self.assertEqual(parent_cfg.child.extra, 9)

    def test_unknown_kwarg_raises(self):
        @override("strict_kw", target=ComponentA.Config)
        def strict_kw(cfg: ComponentA.Config, *, extra: int) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=extra)

        # A kwarg the factory does not accept is a plain TypeError from the call.
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            apply_overrides(
                OverrideConfig(imports=[(__name__, {"typo": 1})]),
                ParentComponent.Config(),
            )

    def test_kwargs_for_no_matching_override_raises(self):
        # kwargs that activate no override is a misconfiguration, not a silent
        # no-op (the registry is empty here after setUp's clear_overrides).
        with self.assertRaisesRegex(ValueError, "activated no override"):
            apply_overrides(
                OverrideConfig(imports=[("torchtitan.config.override", {"x": 1})]),
                ParentComponent.Config(),
            )

    def test_parse_cli_imports(self):
        from torchtitan.config.override import parse_cli_imports

        # Plain modules (comma- or space-separated) and modules whose kwargs are
        # attached to the name as ``module=<json>`` -- the CLI grammar backing
        # ``--override.imports``.
        self.assertEqual(parse_cli_imports(["a.b,c.d"]), ["a.b", "c.d"])
        self.assertEqual(
            parse_cli_imports(['mod={"block_size": 256, "flag": null}']),
            [("mod", {"block_size": 256, "flag": None})],
        )
        self.assertEqual(
            parse_cli_imports(["plain", 'mod={"x": 1}']),
            ["plain", ("mod", {"x": 1})],
        )
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            parse_cli_imports(["mod=123"])


class TestDerive(unittest.TestCase):
    def test_copies_shared_and_applies_deltas(self):
        src = ComponentA.Config(dim=99)
        out = derive(src, DerivedComponent.Config, block_size=256)
        self.assertIsInstance(out, DerivedComponent.Config)
        self.assertEqual(out.dim, 99)  # shared field copied (not in deltas)
        self.assertEqual(out.block_size, 256)  # delta

    def test_target_only_field_falls_back_to_default(self):
        src = ComponentA.Config(dim=7)
        out = derive(src, DerivedComponent.Config)  # no delta for block_size
        self.assertEqual(out.dim, 7)
        self.assertEqual(out.block_size, 128)  # target's declared default

    def test_newly_added_field_carried_without_factory_change(self):
        # ``dim`` stands in for a field the factory never mentions; derive
        # carries the caller's value instead of reverting to the default (64).
        src = ComponentA.Config(dim=4096)
        out = derive(src, DerivedComponent.Config, block_size=64)
        self.assertEqual(out.dim, 4096)

    def test_unknown_delta_raises(self):
        src = ComponentA.Config()
        with self.assertRaisesRegex(ValueError, "not on"):
            derive(src, DerivedComponent.Config, nonexistent=1)

    def test_src_only_fields_dropped(self):
        # ComponentB.Config has {dim, extra}; target ComponentA.Config has {dim}.
        src = ComponentB.Config(dim=5, extra=999)
        out = derive(src, ComponentA.Config)
        self.assertIsInstance(out, ComponentA.Config)
        self.assertEqual(out.dim, 5)
        self.assertFalse(hasattr(out, "extra"))  # src-only field dropped

    def test_non_dataclass_target_raises(self):
        with self.assertRaises(TypeError):
            derive(ComponentA.Config(), object)


class _Model(Configurable):
    """Stand-in for a model config (a `Configurable.Config` with components)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        block: ComponentA.Config = field(default_factory=ComponentA.Config)

    def __init__(self, config: Config):
        self.config = config


class _TrainerCfgHolder(Configurable):
    """Stand-in for Trainer.Config holding a ModelSpec under `model_spec`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        model_spec: object = None

    def __init__(self, config: Config):
        self.config = config


def _make_spec():
    from torchtitan.protocols.model_spec import ModelSpec

    return ModelSpec(
        name="m",
        flavor="f",
        model=_Model.Config(),
        parallelize_fn=lambda: None,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


class TestModelSpecTraversal(unittest.TestCase):
    """Overrides reach the model via ModelSpec.traverse with full-path FQNs."""

    def setUp(self):
        clear_overrides()

    def tearDown(self):
        clear_overrides()

    def test_fqns_are_full_path_from_root(self):
        root = _TrainerCfgHolder.Config(model_spec=_make_spec())
        model_fqns = [fqn for fqn, *_ in root.traverse(_Model.Config)]
        comp_fqns = [fqn for fqn, *_ in root.traverse(ComponentA.Config)]
        # The model config and its components keep the path from the root.
        self.assertEqual(model_fqns, ["model_spec.model"])
        self.assertEqual(comp_fqns, ["model_spec.model.block"])

    def test_component_override_through_model_spec(self):
        @override("blk", target=ComponentA.Config)
        def blk(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim)

        spec = _make_spec()
        root = _TrainerCfgHolder.Config(model_spec=spec)
        replacements = apply_overrides(OverrideConfig(imports=[__name__]), root)
        self.assertEqual(len(replacements), 1)
        self.assertIsInstance(spec.model.block, ComponentB.Config)

    def test_whole_model_vs_component_conflict_detected(self):
        # Regression: a whole-model override (FQN "model_spec.model") and a
        # component override ("model_spec.model.block") must be flagged as an
        # ancestor conflict, not silently lose the component override.
        @override("whole_model", target=_Model.Config)
        def whole(cfg: _Model.Config) -> _Model.Config:
            return _Model.Config()

        @override("blk", target=ComponentA.Config)
        def blk(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim)

        root = _TrainerCfgHolder.Config(model_spec=_make_spec())
        with self.assertRaisesRegex(ValueError, "ancestor"):
            apply_overrides(OverrideConfig(imports=[__name__]), root)


if __name__ == "__main__":
    unittest.main()
