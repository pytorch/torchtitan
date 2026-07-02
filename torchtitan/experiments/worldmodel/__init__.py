def model_registry(*args, **kwargs):
    from .model_config import model_registry as _model_registry

    return _model_registry(*args, **kwargs)


def worldmodel(*args, **kwargs):
    from .config_registry import worldmodel as _worldmodel

    return _worldmodel(*args, **kwargs)

__all__ = ["model_registry", "worldmodel"]
