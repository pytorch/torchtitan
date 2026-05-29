"""Held-out eval entrypoint (the quality measurement) — GPU/torchrun.

Trains the candidate to the eval horizon, then evaluates on the held-out C4 split
at the reference sequence length and prints one parseable line:

    EVAL: eval_loss=<value>

`SubprocessExecutor.run_eval` parses that line. Quality is eval loss (lower is
better); the gate floors it against the golden. This entrypoint is GPU-only and
cannot run in a CPU environment; the orchestration that calls it is tested with
`FakeExecutor`.

NOTE: the exact horizon and the validator's eval-loss accessor must be confirmed
against the live TorchTitan validator API before first GPU use; the wiring below
is the intended shape.
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from torchtitan_autoresearch.verify_config import split_verify_args


def main() -> int:
    # Reuse the verify arg splitter only to strip any --verify.* the caller adds;
    # all remaining args go to TorchTitan's parser unchanged.
    _, titan_argv = split_verify_args([a for a in sys.argv[1:] if a.startswith("--verify.")] +
                                      [a for a in sys.argv[1:] if not a.startswith("--verify.")])
    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args(titan_argv)
    trainer = config.build()

    # Train to the eval horizon (training.steps from the candidate command), then
    # run held-out validation. The validator is configured by the constitution's
    # eval section (held-out C4 at the reference seq).
    trainer.train()
    eval_loss = _run_validation(trainer)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(f"EVAL: eval_loss={eval_loss:.6f}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _run_validation(trainer) -> float:
    """Run the held-out validator and return mean eval loss.

    TODO: confirm the validator's return/accessor on the live API. The validator
    is built from config.validator; `validate()` runs the held-out pass. We read
    the mean eval loss it produces.
    """
    validator = getattr(trainer, "validator", None)
    if validator is None:
        raise RuntimeError("no validator configured; constitution requires a held-out eval")
    result = validator.validate(trainer.model_parts, trainer.step)
    # Accept either a returned scalar/dict or a known attribute.
    if isinstance(result, dict) and "eval_loss" in result:
        return float(result["eval_loss"])
    if isinstance(result, (int, float)):
        return float(result)
    return float(getattr(validator, "last_eval_loss"))


if __name__ == "__main__":
    sys.exit(main())
