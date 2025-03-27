from typing import Any, Mapping, Protocol


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict, returning the updated dict.
    """

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        pass
