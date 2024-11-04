from torch._utils import _get_available_device_type, _get_device_module

DEVICE_TYPE = _get_available_device_type()
DEVICE_MODULE = _get_device_module(DEVICE_TYPE)
