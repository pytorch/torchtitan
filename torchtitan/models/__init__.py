# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import os
import pkgutil
from typing import Dict, Set

import torchtitan.models as models
from torchtitan.model_spec import ModelSpec


_model_specs_path: Set[str] = set()


def _load_module(path: str):
    path = os.path.expanduser(path)

    # 1. Check if path is an existing file or directory path.
    if os.path.exists(path):
        if os.path.isdir(path):
            init_file = os.path.join(path, "__init__.py")
            if os.path.isfile(init_file):
                return _load_module_from_init(path)

            raise ImportError(
                f"Directory '{path}' is not a Python package because it does not "
                "contain an __init__.py file."
            )
        else:
            raise ImportError(f"Path '{path}' is not a directory.")

    # 2. If not a valid path, assume it's a dotted module name.
    return importlib.import_module(path)


def _load_module_from_init(path: str):
    module_name = os.path.basename(os.path.normpath(path))
    init_file = os.path.join(path, "__init__.py")

    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if spec is None:
        raise ImportError(f"Could not create spec from '{init_file}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for _, name, _ in pkgutil.iter_modules(models.__path__):
    full_module_name = f"{models.__name__}.{name}"
    _model_specs_path.add(full_module_name)
    # model_module = importlib.import_module(full_module_name)
    # load_spec_from_module(model_module)


def add_model_spec_path(path: str):
    global _model_specs_path
    _model_specs_path.add(path)


def build_model_specs() -> Dict[str, ModelSpec]:
    """
    Load all model specs from the `models` package.
    """
    global _model_specs_path
    model_specs = {}
    for path in _model_specs_path:
        module = _load_module(path)
        model_spec = getattr(module, "model_spec", None)
        if model_spec is not None:
            model_specs[model_spec.name] = model_spec
        # We would like to just use `model_spec` but current torchtitan parallelize
        # functions depend on ModelArgs and can cause circular imports.
        # As a result, we have to use `build_model_spec` as a workaround.
        build_model_spec = getattr(module, "build_model_spec", None)
        if build_model_spec:
            model_spec = build_model_spec()
            model_specs[model_spec.name] = model_spec

    return model_specs


__all__ = [add_model_spec_path, build_model_specs]
