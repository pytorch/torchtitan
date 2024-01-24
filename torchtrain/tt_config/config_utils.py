import pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

def get_config(config_path: str = "./torchtrain/tt_config/train_config.toml") -> dict:
    """
    Reads a config file in TOML format and returns a dictionary.
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config
