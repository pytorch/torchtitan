import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils

from cutlass.cute.runtime import from_dlpack

cluster_shape_mn = (4, 4)

hardware_info = utils.HardwareInfo()
print(f"hardware_info: {hardware_info}")
max_active_clusters = hardware_info.get_max_active_clusters(
    cluster_shape_mn[0] * cluster_shape_mn[1]
)

print(f"max_active_clusters: {max_active_clusters}")
