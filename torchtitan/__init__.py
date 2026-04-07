from importlib.metadata import version

try:
    __version__ = version("torchtitan")
except Exception:
    __version__ = "0.0.0+unknown"
