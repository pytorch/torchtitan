from ._dstorage import DStorage, fully_shard_flat, get_dstorage, Owned
from ._patch_dtensor import patch_dtensor_placements, register_custom_placement

# Register Owned placement and apply the monkey-patch
register_custom_placement(Owned)
patch_dtensor_placements()


__all__ = [
    "DStorage",
    "fully_shard_flat",
    "get_dstorage",
    "Owned",
    "patch_dtensor_placements",
    "register_custom_placement",
]
