"""Unit tests for InterleavedDataset."""
import unittest
from copy import deepcopy

from torch.utils.data import IterableDataset as TorchIterableDataset

from torchtitan.hf_datasets.interleave import InterleavedDataset


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _MockDataset(TorchIterableDataset):
    """Minimal stateful iterable dataset for testing."""

    def __init__(self, values: list):
        self._values = list(values)
        self._pos: int = 0

    def __iter__(self):
        while self._pos < len(self._values):
            val = self._values[self._pos]
            self._pos += 1
            yield val
        self._pos = 0

    def state_dict(self):
        return {"pos": self._pos}

    def load_state_dict(self, sd: dict):
        self._pos = sd["pos"]


class _NoCheckpointDataset:
    """Missing state_dict / load_state_dict — must be rejected by InterleavedDataset."""

    def __iter__(self):
        yield from range(5)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInterleavedDatasetInit(unittest.TestCase):

    def test_rejects_missing_state_dict(self):
        with self.assertRaises(TypeError):
            InterleavedDataset([_NoCheckpointDataset()], [1.0])

    def test_rejects_length_mismatch(self):
        ds = _MockDataset([1])
        with self.assertRaises(ValueError):
            InterleavedDataset([ds, ds], [1.0])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            InterleavedDataset([], [])

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            InterleavedDataset([_MockDataset([1])], [-1.0])

    def test_rejects_zero_sum_weights(self):
        with self.assertRaises(ValueError):
            InterleavedDataset([_MockDataset([1])], [0.0])

    def test_normalizes_weights(self):
        ds_a, ds_b = _MockDataset([1]), _MockDataset([2])
        interleaved = InterleavedDataset([ds_a, ds_b], [2.0, 8.0])
        self.assertAlmostEqual(sum(interleaved._probs), 1.0)
        self.assertAlmostEqual(interleaved._probs[0], 0.2)
        self.assertAlmostEqual(interleaved._probs[1], 0.8)


# ---------------------------------------------------------------------------
# Iteration behaviour
# ---------------------------------------------------------------------------

class TestInterleavedDatasetIteration(unittest.TestCase):

    def test_single_source_yields_all_samples(self):
        ds = _MockDataset(list(range(10)))
        interleaved = InterleavedDataset([ds], [1.0], seed=0)
        self.assertEqual(sorted(list(interleaved)), list(range(10)))

    def test_stops_on_first_source_exhaustion(self):
        """When one source runs out, iteration stops immediately even if
        others still have data."""
        ds_short = _MockDataset([99])              # 1 item
        ds_long  = _MockDataset(list(range(50)))   # 50 items
        samples = list(InterleavedDataset([ds_short, ds_long], [1.0, 1.0], seed=0))

        self.assertIn(99, samples)
        # Long source is cut short
        long_samples = [v for v in samples if v != 99]
        self.assertGreater(len(long_samples), 0)
        self.assertLess(len(long_samples), 50)

    def test_sampling_respects_weight_ratio(self):
        """Higher-weighted source is drawn proportionally more often."""
        # Use distinct value ranges to identify source without source_idx
        ds_a = _MockDataset(list(range(1000)))           # values 0–999
        ds_b = _MockDataset(list(range(1000, 2000)))     # values 1000–1999
        samples = list(InterleavedDataset([ds_a, ds_b], [1.0, 9.0], seed=7))

        count_a = sum(1 for v in samples if v < 1000)
        count_b = sum(1 for v in samples if v >= 1000)
        # ds_a (lower weight) exhausts first; draw ratio should be roughly 1:9
        self.assertGreater(count_b / count_a, 6.0)  # generous tolerance

    def test_deterministic_with_same_seed(self):
        def run(seed):
            ds_a = _MockDataset(list(range(20)))
            ds_b = _MockDataset(list(range(20, 40)))
            return list(InterleavedDataset([ds_a, ds_b], [1.0, 1.0], seed=seed))

        self.assertEqual(run(42), run(42))
        self.assertNotEqual(run(42), run(99))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class TestInterleavedDatasetCheckpointing(unittest.TestCase):

    def test_state_dict_keys(self):
        ds = _MockDataset(list(range(5)))
        sd = InterleavedDataset([ds], [1.0], seed=0).state_dict()
        for key in ("rng_state", "sources"):
            self.assertIn(key, sd)
        self.assertEqual(len(sd["sources"]), 1)

    def test_source_state_delegated_to_child(self):
        """sources[i] is exactly what child.state_dict() returns."""
        ds = _MockDataset(list(range(5)))
        interleaved = InterleavedDataset([ds], [1.0], seed=0)
        it = iter(interleaved)
        next(it)
        sd = interleaved.state_dict()
        self.assertEqual(sd["sources"][0], ds.state_dict())

    def test_resume_produces_same_tail(self):
        """Restoring state mid-iteration continues with identical sample order."""
        ds_a = _MockDataset(list(range(10)))
        ds_b = _MockDataset(list(range(10, 20)))
        interleaved = InterleavedDataset([ds_a, ds_b], [1.0, 1.0], seed=42)

        it = iter(interleaved)
        for _ in range(6):
            next(it)
        state = deepcopy(interleaved.state_dict())
        tail_original = list(it)

        ds_a2 = _MockDataset(list(range(10)))
        ds_b2 = _MockDataset(list(range(10, 20)))
        interleaved2 = InterleavedDataset([ds_a2, ds_b2], [1.0, 1.0], seed=42)
        interleaved2.load_state_dict(state)
        self.assertEqual(list(interleaved2), tail_original)

    def test_resume_restores_rng_state(self):
        """After load_state_dict, the interleaver RNG produces the same sequence."""
        ds_a = _MockDataset(list(range(20)))
        ds_b = _MockDataset(list(range(20, 40)))
        interleaved = InterleavedDataset([ds_a, ds_b], [1.0, 1.0], seed=13)

        it = iter(interleaved)
        for _ in range(8):
            next(it)
        state = deepcopy(interleaved.state_dict())
        original_tail = list(it)

        ds_a2 = _MockDataset(list(range(20)))
        ds_b2 = _MockDataset(list(range(20, 40)))
        interleaved2 = InterleavedDataset([ds_a2, ds_b2], [1.0, 1.0], seed=13)
        interleaved2.load_state_dict(state)
        self.assertEqual(list(interleaved2), original_tail)


if __name__ == "__main__":
    unittest.main()
