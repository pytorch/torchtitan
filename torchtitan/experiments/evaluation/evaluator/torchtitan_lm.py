# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import List, Tuple

import torch
import torch.distributed.distributed_c10d as c10d

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model


def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(c10d.get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
# Refer to https://github.com/facebookresearch/lingua/blob/main/apps/main/eval.py
@register_model("torchtitan", "titan", "titan_lm")
class TorchTitanLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        # TODO: Only supports single GPU for now
        self._rank = 0
        self._world_size = 1
        # self._rank = c10d.get_rank()
        # self._world_size = c10d.get_world_size()
        self.device = generator.device

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(self.generator.tokenizer.encode(p, bos=False, eos=False))
            p_len -= 1  # TODO: Adjust for the BOS token
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(), len(ll)))
        self.generator.max_gen_len = max_gen_len

        return results