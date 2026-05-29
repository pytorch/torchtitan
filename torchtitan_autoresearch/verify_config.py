"""Parsing for the harness's own ``--verify.*`` flags.

TorchTitan parses its config with a strict ``tyro`` CLI that rejects unknown
arguments, so the harness must strip its own flags before handing the rest to
``ConfigManager.parse_args``. ``split_verify_args`` does exactly that and never
mutates the order of the remaining TorchTitan arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default gradient tensors to fingerprint. Selection is by substring with a
# largest-2D-parameter fallback (see compare.select_param_names), so these work
# even if a candidate renames internal modules.
DEFAULT_PARAM_PATTERNS = ("lm_head", "attention.wq", "feed_forward.w1")


@dataclass
class VerifyArgs:
    """Harness-only options controlling capture vs. compare behavior.

    ``mode`` is ``capture`` to write a golden/champion snapshot or ``compare``
    to gate a candidate. ``cls`` declares the change class so the comparison can
    apply the matching tolerance preset and audit the agent's own claim.
    """

    mode: str = "compare"  # "capture" | "compare"
    snapshot: str = ""  # capture: output path; compare: golden (high-precision) path
    champion: str = ""  # compare: optional same-config incremental-anchor path
    cls: str = "precision"  # "schedule" | "reduction" | "precision" | "kernel"
    batch: str = "data"  # "data" (seeded c4_test batch 0) | "stress" (synthetic)
    seed: int = 42
    proj_dim: int = 4096  # feature-hashing sketch width for gradient direction
    n_logit_samples: int = 256
    result_json: str = ""  # optional path to write the machine-readable verdict
    param_patterns: tuple[str, ...] = field(default=DEFAULT_PARAM_PATTERNS)


_BOOL_KEYS: set[str] = set()  # no boolean verify flags today; reserved for growth


def split_verify_args(argv: list[str]) -> tuple[VerifyArgs, list[str]]:
    """Pull ``--verify.*`` flags out of ``argv``; return (VerifyArgs, rest).

    Supports both ``--verify.key=value`` and ``--verify.key value`` spellings.
    Everything not prefixed with ``--verify.`` is preserved verbatim so it can
    be forwarded to TorchTitan's own parser unchanged.
    """
    args = VerifyArgs()
    rest: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--verify."):
            if "=" in tok:
                key, val = tok[len("--verify.") :].split("=", 1)
            else:
                key = tok[len("--verify.") :]
                if key in _BOOL_KEYS:
                    val = "true"
                else:
                    i += 1
                    val = argv[i] if i < len(argv) else ""
            _assign(args, key, val)
        else:
            rest.append(tok)
        i += 1
    _validate(args)
    return args, rest


def _assign(args: VerifyArgs, key: str, val: str) -> None:
    if not hasattr(args, key):
        raise ValueError(f"unknown --verify.{key}")
    if key in ("seed", "proj_dim", "n_logit_samples"):
        setattr(args, key, int(val))
    elif key == "param_patterns":
        setattr(args, key, tuple(p for p in val.split(",") if p))
    else:
        setattr(args, key, val)


def _validate(args: VerifyArgs) -> None:
    if args.mode not in ("capture", "compare"):
        raise ValueError(f"--verify.mode must be capture|compare, got {args.mode!r}")
    if args.cls not in ("schedule", "reduction", "precision", "kernel"):
        raise ValueError(f"--verify.cls invalid: {args.cls!r}")
    if args.batch not in ("data", "stress"):
        raise ValueError(f"--verify.batch must be data|stress, got {args.batch!r}")
    if not args.snapshot:
        raise ValueError("--verify.snapshot is required (capture target or golden)")
