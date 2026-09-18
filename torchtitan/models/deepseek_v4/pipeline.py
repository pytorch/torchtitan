# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    pipeline_llm,
)


def pipeline_deepseek_v4(model, *, parallel_dims, parallelism, model_config, **kwargs):
    """Keep the HC reduction with the decoder's final normalization."""
    module_fqns = parallelism.module_fqns_per_model_part
    if module_fqns is None:
        module_fqns = _generate_llm_fqn_per_model_part(
            *_get_pipeline_metadata(parallel_dims, parallelism, model_config)
        )
        module_fqns[-1].insert(module_fqns[-1].index("norm"), "hc_head")
        parallelism = replace(parallelism, module_fqns_per_model_part=module_fqns)
    else:
        hc_stages = [i for i, names in enumerate(module_fqns) if "hc_head" in names]
        norm_stages = [i for i, names in enumerate(module_fqns) if "norm" in names]
        if len(hc_stages) != 1 or hc_stages != norm_stages:
            raise ValueError(
                "DeepSeek V4 pipeline splits must place hc_head and norm "
                "together on exactly one stage."
            )

    return pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )
