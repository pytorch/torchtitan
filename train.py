"""
MoE model training entry point.

Usage:
    torchrun --nproc_per_node=8 train.py --config configs/moe_tiny.yaml
"""

import typer
import yaml

from src.config.config import Config
from src.trainer import train

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
):
    with open(config) as f:
        cfg = Config(**(yaml.safe_load(f) or {}))
    train(cfg)


if __name__ == "__main__":
    app()
