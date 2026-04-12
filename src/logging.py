import logfire

logfire.configure(
    service_name="moe-training",
    send_to_logfire="if-token-present",
    inspect_arguments=False,
)

logger = logfire


def init_logger() -> None:
    """No-op — logfire is configured at import time."""
    pass
