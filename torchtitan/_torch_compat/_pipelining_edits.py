# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact function edits from PyTorch PRs #199884, #200238, and #200199.

Targets the official PyTorch 2.15.0 RC, installed as torch==2.15.0 from
https://download.pytorch.org/whl/test/cu130 for the CUDA 13.0 variant.

The validated wheel reports 2.15.0+cu130, with source commit
dd1fefe31baf1bc8efbab6dc5291eebfb948b7bf. Only changed functions are patched;
unchanged PyTorch implementation stays in the installed wheel.
"""

from typing import NotRequired, TypedDict


class FunctionEdit(TypedDict):
    target: str
    source: NotRequired[str]
    before_sha256: NotRequired[str]
    after_sha256: NotRequired[str]
    edits: NotRequired[list[tuple[str, str]]]


class Backport(TypedDict):
    pr: int
    module: str
    functions: list[FunctionEdit]


PATCHES: list[Backport] = [
    {
        "pr": 199884,
        "module": "torch.distributed.pipelining.schedules",
        "functions": [
            {
                "target": "_PipelineSchedule.__init__",
                "before_sha256": "7111a19ed694da1d20b5d5951e08aeb16e0d7220fc44384dd8e8ff3685b7354e",
                "after_sha256": "0a734cf5c711d009d4c131c5ae3a050b300f1f0c4c29a02fc491e25ba3b17e3c",
                "edits": [
                    (
                        "    self._has_backward = self._loss_fn is not None\n"
                        "    self._p2p_initialized = False\n"
                        "\n"
                        "    # Holds the losses for each microbatch.\n",
                        "    self._has_backward = self._loss_fn is not None\n"
                        "    self._p2p_initialized = False\n"
                        "    self._post_metadata_inference_cleanups: OrderedDict[int, "
                        "Callable[[], None]] = (\n"
                        "        OrderedDict()\n"
                        "    )\n"
                        "\n"
                        "    # Holds the losses for each microbatch.\n",
                    )
                ],
            },
            {
                "target": "_PipelineSchedule.register_post_metadata_inference_cleanup",
                "source": "def register_post_metadata_inference_cleanup(\n"
                "    self, callback: Callable[[], None]\n"
                ") -> RemovableHandle:\n"
                '    """Register a callback after dynamic metadata inference is '
                "cleaned up.\n"
                "\n"
                "    The callback runs after all local stages restore their "
                "metadata-inference\n"
                "    state. It does not run when stages use constructor-provided "
                "static metadata.\n"
                "\n"
                "    Args:\n"
                "        callback: Zero-argument cleanup called after dynamic "
                "metadata inference.\n"
                "\n"
                "    Returns:\n"
                "        A handle that removes the callback when no longer needed.\n"
                '    """\n'
                "    handle = "
                "RemovableHandle(self._post_metadata_inference_cleanups)\n"
                "    self._post_metadata_inference_cleanups[handle.id] = callback\n"
                "    return handle\n",
            },
            {
                "target": "_PipelineSchedule._initialize_pp_stages",
                "before_sha256": "ca3fab1f7f73cae788e0c91cbb2a2c026d7e00f4ecc0411363921c5416b227ac",
                "after_sha256": "89a7e38bf5f969a8fbb5811f5f5385a8686fffc1b691cb0dacdb3fbe528cb46c",
                "edits": [
                    (
                        "        stage for stage in stages if isinstance(stage, "
                        "PipelineStage)\n"
                        "    ]\n"
                        "    for stage in pipeline_stages:\n"
                        "        stage._pre_metadata_inference_backup()\n",
                        "        stage for stage in stages if isinstance(stage, "
                        "PipelineStage)\n"
                        "    ]\n"
                        "    uses_dynamic_metadata = any(\n"
                        "        stage._inference_mode == InferenceMode.DYNAMIC for stage "
                        "in pipeline_stages\n"
                        "    )\n"
                        "    for stage in pipeline_stages:\n"
                        "        stage._pre_metadata_inference_backup()\n",
                    ),
                    (
                        "        for stage in pipeline_stages:\n"
                        "            stage._post_metadata_inference_cleanup()\n"
                        "\n"
                        "    return fwd_initialized, bwd_initialized\n",
                        "        for stage in pipeline_stages:\n"
                        "            stage._post_metadata_inference_cleanup()\n"
                        "        if uses_dynamic_metadata:\n"
                        "            for cleanup in "
                        "tuple(self._post_metadata_inference_cleanups.values()):\n"
                        "                cleanup()\n"
                        "\n"
                        "    return fwd_initialized, bwd_initialized\n",
                    ),
                ],
            },
        ],
    },
    {
        "pr": 200238,
        "module": "torch.distributed.pipelining.schedules",
        "functions": [
            {
                "target": "_PipelineSchedule.__init__",
                "before_sha256": "0a734cf5c711d009d4c131c5ae3a050b300f1f0c4c29a02fc491e25ba3b17e3c",
                "after_sha256": "6f2e25fa3a5a85a82046ea269ba528985b106850fe1fd125b074093e43b34a89",
                "edits": [
                    (
                        "    # Holds the losses for each microbatch.\n"
                        "    self._internal_losses: list[torch.Tensor] = []\n"
                        '    logger.info("Using %s", self.__class__.__name__)\n',
                        "    # Holds the losses for each microbatch.\n"
                        "    self._internal_losses: list[torch.Tensor] = []\n"
                        "    # Per-microbatch loss keyword arguments of the current step. "
                        "Set by\n"
                        "    # ``step``.\n"
                        "    self._loss_kwarg_mbs: list[dict[str, Any]] | None = None\n"
                        '    logger.info("Using %s", self.__class__.__name__)\n',
                    )
                ],
            },
            {
                "target": "_PipelineSchedule._maybe_compute_loss",
                "before_sha256": "85dcc310457ffceef5081629e69a7856807a960e782133f7d1abce3079a5f883",
                "after_sha256": "380948980a0ed38de4127bb29f78545d06f098ad0fbecf5f9f8add5ca6f0e722",
                "edits": [
                    (
                        "):\n"
                        "    if stage.is_last and self._loss_fn is not None:\n"
                        "        loss = self._compute_loss(output, target_mbs[mb_index], "
                        "loss_kwargs)  # type: ignore[index]\n"
                        "        self._internal_losses.append(loss)\n",
                        "):\n"
                        "    if stage.is_last and self._loss_fn is not None:\n"
                        "        loss_kwarg_mb = (\n"
                        "            self._loss_kwarg_mbs[mb_index]\n"
                        "            if self._loss_kwarg_mbs is not None\n"
                        "            else None\n"
                        "        )\n"
                        "        loss = self._compute_loss(\n"
                        "            output,\n"
                        "            target_mbs[mb_index],  # type: ignore[index]\n"
                        "            loss_kwargs,\n"
                        "            loss_kwarg_mb,\n"
                        "        )\n"
                        "        self._internal_losses.append(loss)\n",
                    )
                ],
            },
            {
                "target": "_PipelineSchedule._initialize_pp_stages",
                "before_sha256": "89a7e38bf5f969a8fbb5811f5f5385a8686fffc1b691cb0dacdb3fbe528cb46c",
                "after_sha256": "5c353acc08642e3b4afe49b2a86a5c5a35831e1f08ce246eabbaac72c223ad81",
                "edits": [
                    (
                        "\n"
                        "            if needs_bwd:\n"
                        "                prev_stage_grad_meta: Any = None\n"
                        "                for stage in reversed(stages):\n",
                        "\n"
                        "            if needs_bwd:\n"
                        "                # Backward metadata inference runs the loss on "
                        "the first\n"
                        "                # microbatch's target and loss keyword "
                        "arguments.\n"
                        "                if self._loss_kwarg_mbs is not None:\n"
                        "                    loss_kwargs = {\n"
                        "                        **(loss_kwargs or {}),\n"
                        "                        **self._loss_kwarg_mbs[0],\n"
                        "                    }\n"
                        "                prev_stage_grad_meta: Any = None\n"
                        "                for stage in reversed(stages):\n",
                    )
                ],
            },
            {
                "target": "_PipelineSchedule.step",
                "before_sha256": "5d83690748ce9bb83a5445f4536d4adc5a978a5fb5428520c3c5d183f2667e46",
                "after_sha256": "02e3a8be40759eb935c0893ec98d6b1923ff868c6907a35520b5ca53f506f3f2",
                "edits": [
                    (
                        "    kwarg_mbs: Any = None,\n"
                        "    target_mbs: Any = None,\n"
                        "    finalize_gradients: bool = True,\n"
                        "    **kwargs,\n",
                        "    kwarg_mbs: Any = None,\n"
                        "    target_mbs: Any = None,\n"
                        "    loss_kwarg_mbs: list[dict[str, Any]] | None = None,\n"
                        "    finalize_gradients: bool = True,\n"
                        "    **kwargs,\n",
                    )
                ],
            },
            {
                "target": "_PipelineSchedule._check_inputs",
                "before_sha256": "2788bc0c56c1ffc5daa7113437a65a73f9723c4fa446fef6a4bddc7db16c5ca6",
                "after_sha256": "296a7d1f6f0f21e6e187b6a8276efb9f0c23cb7fed364aba0a43d23b8c389130",
                "edits": [
                    (
                        "    target_mbs: list | None = None,\n"
                        "    losses: list | None = None,\n"
                        ") -> tuple[list, list]:\n"
                        '    """\n',
                        "    target_mbs: list | None = None,\n"
                        "    losses: list | None = None,\n"
                        "    loss_kwarg_mbs: list | None = None,\n"
                        ") -> tuple[list, list]:\n"
                        '    """\n',
                    ),
                    (
                        '        check_type_and_len(target_mbs, "target_mbs")\n'
                        "\n"
                        "    if losses is not None:\n"
                        "        if not isinstance(losses, list):\n",
                        '        check_type_and_len(target_mbs, "target_mbs")\n'
                        "\n"
                        "    if loss_kwarg_mbs is not None:\n"
                        '        check_type_and_len(loss_kwarg_mbs, "loss_kwarg_mbs")\n'
                        "\n"
                        "    if losses is not None:\n"
                        "        if not isinstance(losses, list):\n",
                    ),
                ],
            },
            {
                "target": "_PipelineSchedule._compute_loss",
                "before_sha256": "58c00d36677888b2d1e7519a9c16b74644409d6497a3bb8787a7e19db316ffcc",
                "after_sha256": "199a4d8619e30523c8e04adfe91d95387f436b8471b178036be735e71fbce488",
                "edits": [
                    (
                        "def _compute_loss(self, output, target, loss_kwargs=None):\n"
                        "    return self._loss_fn(output, target, **(loss_kwargs or {}))  "
                        "# type: ignore[misc]\n",
                        "def _compute_loss(self, output, target, loss_kwargs=None, "
                        "loss_kwarg_mb=None):\n"
                        "    return self._loss_fn(  # type: ignore[misc]\n"
                        "        output, target, **(loss_kwargs or {}), **(loss_kwarg_mb "
                        "or {})\n"
                        "    )\n",
                    )
                ],
            },
            {
                "target": "_PipelineSchedule._get_microbatch_inputs",
                "before_sha256": "2d3eec0d454e0834351c9dbf32da23d2918cd4d43ba2f74144795670d2dcae3d",
                "after_sha256": "f8261690b20ef445cfd5513f428ffb60a91824734f88004b9c47d3d25e9b916a",
                "edits": [
                    (
                        "    kwarg_mbs: Any,\n"
                        "    target_mbs: Any,\n"
                        ") -> tuple[list | None, list | None, list | None]:\n"
                        "    pre_split = any(mbs is not None for mbs in (arg_mbs, "
                        "kwarg_mbs, target_mbs))\n"
                        "    if not pre_split:\n"
                        "        args_split, kwargs_split = self._split_inputs(args, "
                        "kwargs)\n",
                        "    kwarg_mbs: Any,\n"
                        "    target_mbs: Any,\n"
                        "    loss_kwarg_mbs: Any = None,\n"
                        ") -> tuple[list | None, list | None, list | None]:\n"
                        "    pre_split = any(\n"
                        "        mbs is not None for mbs in (arg_mbs, kwarg_mbs, "
                        "target_mbs, loss_kwarg_mbs)\n"
                        "    )\n"
                        "    if not pre_split:\n"
                        "        args_split, kwargs_split = self._split_inputs(args, "
                        "kwargs)\n",
                    ),
                    (
                        "        )\n"
                        "\n"
                        "    arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, "
                        "target_mbs)\n"
                        "\n"
                        "    for mb_index, (arg_mb, kwarg_mb) in enumerate(\n",
                        "        )\n"
                        "\n"
                        "    arg_mbs, kwarg_mbs = self._check_inputs(\n"
                        "        arg_mbs, kwarg_mbs, target_mbs, "
                        "loss_kwarg_mbs=loss_kwarg_mbs\n"
                        "    )\n"
                        "\n"
                        "    for mb_index, (arg_mb, kwarg_mb) in enumerate(\n",
                    ),
                    (
                        "            )\n\n    return arg_mbs, kwarg_mbs, target_mbs\n",
                        "            )\n"
                        "\n"
                        "    for mb_index, loss_kwarg_mb in enumerate(loss_kwarg_mbs or "
                        "()):\n"
                        "        if not isinstance(loss_kwarg_mb, dict):\n"
                        "            raise TypeError(\n"
                        '                "loss_kwarg_mbs must be a list of dicts, but "\n'
                        '                f"loss_kwarg_mbs[{mb_index}] is a '
                        '{type(loss_kwarg_mb)}"\n'
                        "            )\n"
                        "\n"
                        "    return arg_mbs, kwarg_mbs, target_mbs\n",
                    ),
                ],
            },
            {
                "target": "PipelineScheduleSingle.step",
                "before_sha256": "e5950226953c4bbe41f66120843f26c33c2a3aeb380466ea902cdf1a5cb91820",
                "after_sha256": "0a3d09c0c74abccbcd18b06bf0fd8ae6053b75ca5c0eb4c4ee8dcf31b302f5bf",
                "edits": [
                    (
                        "    kwarg_mbs: Any = None,\n"
                        "    target_mbs: Any = None,\n"
                        "    finalize_gradients: bool = True,\n"
                        "    **kwargs,\n",
                        "    kwarg_mbs: Any = None,\n"
                        "    target_mbs: Any = None,\n"
                        "    loss_kwarg_mbs: list[dict[str, Any]] | None = None,\n"
                        "    finalize_gradients: bool = True,\n"
                        "    **kwargs,\n",
                    ),
                    (
                        "        return_outputs (bool, optional): Whether to merge and "
                        "return output\n"
                        "            chunks on the last stage. Default: ``True``.\n"
                        "        loss_kwargs (dict, optional): Extra keyword arguments "
                        "forwarded to\n"
                        "            the configured ``loss_fn``. Default: ``None``.\n"
                        "        arg_mbs (list[tuple], optional): Pre-split positional "
                        "inputs, one\n"
                        "            tuple per microbatch. Default: ``None``.\n",
                        "        return_outputs (bool, optional): Whether to merge and "
                        "return output\n"
                        "            chunks on the last stage. Default: ``True``.\n"
                        "        loss_kwargs (dict, optional): Keyword arguments passed "
                        "unchanged\n"
                        "            to the configured ``loss_fn`` for every microbatch. "
                        "Unlike\n"
                        "            ``kwargs``, they are not split. Default: ``None``.\n"
                        "        arg_mbs (list[tuple], optional): Pre-split positional "
                        "inputs, one\n"
                        "            tuple per microbatch. Default: ``None``.\n",
                    ),
                    (
                        "        target_mbs (list, optional): Pre-split targets, one entry "
                        "per\n"
                        "            microbatch. Default: ``None``.\n"
                        "        finalize_gradients (bool, optional): Whether to reduce "
                        "accumulated\n"
                        "            FSDP gradients and reshard FSDP parameters at the end "
                        "of this\n",
                        "        target_mbs (list, optional): Pre-split targets, one entry "
                        "per\n"
                        "            microbatch. Default: ``None``.\n"
                        "        loss_kwarg_mbs (list[dict], optional): Pre-split loss "
                        "keyword\n"
                        "            arguments, one dict per microbatch, such as per-token "
                        "tensors.\n"
                        "            Like the other pre-split inputs, pass targets "
                        "through\n"
                        "            ``target_mbs`` with it. Microbatch ``i`` computes\n"
                        "            ``loss_fn(output, target_mbs[i], **loss_kwargs,\n"
                        "            **loss_kwarg_mbs[i])``, so a key must not appear in "
                        "both. While\n"
                        "            inferring stage metadata (on the first step by "
                        "default), the\n"
                        "            schedule also calls ``loss_fn`` once with microbatch "
                        "0's\n"
                        "            inputs, so ``loss_fn`` must not accumulate into "
                        "objects passed\n"
                        "            this way. Default: ``None``.\n"
                        "        finalize_gradients (bool, optional): Whether to reduce "
                        "accumulated\n"
                        "            FSDP gradients and reshard FSDP parameters at the end "
                        "of this\n",
                    ),
                    (
                        "        kwarg_mbs,\n"
                        "        target_mbs,\n"
                        "    )\n"
                        "\n"
                        "    # Run microbatches\n"
                        "    self._step_microbatches(\n"
                        "        args_split,\n"
                        "        kwargs_split,\n"
                        "        targets_split,\n"
                        "        losses,\n"
                        "        return_outputs,\n"
                        "        loss_kwargs=loss_kwargs,\n"
                        "    )\n"
                        "\n"
                        "    # Return merged results per original format\n",
                        "        kwarg_mbs,\n"
                        "        target_mbs,\n"
                        "        loss_kwarg_mbs,\n"
                        "    )\n"
                        "\n"
                        "    self._loss_kwarg_mbs = loss_kwarg_mbs\n"
                        "    try:\n"
                        "        # Run microbatches\n"
                        "        self._step_microbatches(\n"
                        "            args_split,\n"
                        "            kwargs_split,\n"
                        "            targets_split,\n"
                        "            losses,\n"
                        "            return_outputs,\n"
                        "            loss_kwargs=loss_kwargs,\n"
                        "        )\n"
                        "    finally:\n"
                        "        self._loss_kwarg_mbs = None\n"
                        "\n"
                        "    # Return merged results per original format\n",
                    ),
                ],
            },
            {
                "target": "PipelineScheduleMulti.step",
                "before_sha256": "a00b8d42d61d8e09fd316afa3f1ff88a830a00f028764cfb1eec4bded1d982c7",
                "after_sha256": "b251dfeac673455b0660cf5dc8bb974d94f6b19707f482ba73150e3214fadeb9",
                "edits": [
                    (
                        "    kwarg_mbs: Any = None,\n"
                        "    target_mbs: Any = None,\n"
                        "    finalize_gradients: bool = True,\n"
                        "    **kwargs,\n",
                        "    kwarg_mbs: Any = None,\n"
                        "    target_mbs: Any = None,\n"
                        "    loss_kwarg_mbs: list[dict[str, Any]] | None = None,\n"
                        "    finalize_gradients: bool = True,\n"
                        "    **kwargs,\n",
                    ),
                    (
                        "        return_outputs (bool, optional): Whether to merge and "
                        "return output\n"
                        "            chunks on the last stage. Default: ``True``.\n"
                        "        loss_kwargs (dict, optional): Extra keyword arguments "
                        "forwarded to\n"
                        "            the configured ``loss_fn``. Default: ``None``.\n"
                        "        arg_mbs (list[tuple], optional): Pre-split positional "
                        "inputs, one\n"
                        "            tuple per microbatch. Default: ``None``.\n",
                        "        return_outputs (bool, optional): Whether to merge and "
                        "return output\n"
                        "            chunks on the last stage. Default: ``True``.\n"
                        "        loss_kwargs (dict, optional): Keyword arguments passed "
                        "unchanged\n"
                        "            to the configured ``loss_fn`` for every microbatch. "
                        "Unlike\n"
                        "            ``kwargs``, they are not split. Default: ``None``.\n"
                        "        arg_mbs (list[tuple], optional): Pre-split positional "
                        "inputs, one\n"
                        "            tuple per microbatch. Default: ``None``.\n",
                    ),
                    (
                        "        target_mbs (list, optional): Pre-split targets, one entry "
                        "per\n"
                        "            microbatch. Default: ``None``.\n"
                        "        finalize_gradients (bool, optional): Whether to reduce "
                        "accumulated\n"
                        "            FSDP gradients and reshard FSDP parameters at the end "
                        "of this\n",
                        "        target_mbs (list, optional): Pre-split targets, one entry "
                        "per\n"
                        "            microbatch. Default: ``None``.\n"
                        "        loss_kwarg_mbs (list[dict], optional): Pre-split loss "
                        "keyword\n"
                        "            arguments, one dict per microbatch, such as per-token "
                        "tensors.\n"
                        "            Like the other pre-split inputs, pass targets "
                        "through\n"
                        "            ``target_mbs`` with it. Microbatch ``i`` computes\n"
                        "            ``loss_fn(output, target_mbs[i], **loss_kwargs,\n"
                        "            **loss_kwarg_mbs[i])``, so a key must not appear in "
                        "both. While\n"
                        "            inferring stage metadata (on the first step by "
                        "default), the\n"
                        "            schedule also calls ``loss_fn`` once with microbatch "
                        "0's\n"
                        "            inputs, so ``loss_fn`` must not accumulate into "
                        "objects passed\n"
                        "            this way. Default: ``None``.\n"
                        "        finalize_gradients (bool, optional): Whether to reduce "
                        "accumulated\n"
                        "            FSDP gradients and reshard FSDP parameters at the end "
                        "of this\n",
                    ),
                    (
                        "        kwarg_mbs,\n"
                        "        target_mbs,\n"
                        "    )\n"
                        "\n"
                        "    # Run microbatches\n"
                        "    self._step_microbatches(\n"
                        "        args_split,\n"
                        "        kwargs_split,\n"
                        "        targets_split,\n"
                        "        losses,\n"
                        "        return_outputs,\n"
                        "        loss_kwargs=loss_kwargs,\n"
                        "    )\n"
                        "\n"
                        "    # Return merged results per original format\n",
                        "        kwarg_mbs,\n"
                        "        target_mbs,\n"
                        "        loss_kwarg_mbs,\n"
                        "    )\n"
                        "\n"
                        "    self._loss_kwarg_mbs = loss_kwarg_mbs\n"
                        "    try:\n"
                        "        # Run microbatches\n"
                        "        self._step_microbatches(\n"
                        "            args_split,\n"
                        "            kwargs_split,\n"
                        "            targets_split,\n"
                        "            losses,\n"
                        "            return_outputs,\n"
                        "            loss_kwargs=loss_kwargs,\n"
                        "        )\n"
                        "    finally:\n"
                        "        self._loss_kwarg_mbs = None\n"
                        "\n"
                        "    # Return merged results per original format\n",
                    ),
                ],
            },
        ],
    },
    {
        "pr": 200199,
        "module": "torch.distributed.pipelining._backward",
        "functions": [
            {
                "target": "stage_backward_input",
                "before_sha256": "855638a38edcde06a1b402e40e68e7f898efe68f9ae74a97856c5da83fbd575f",
                "after_sha256": "728971bdb048ebc46ad929c7c54633fcdb602387680ef8c310c5088ab3c83a78",
                "edits": [
                    (
                        "                    inp.grad += dinput\n"
                        "\n"
                        "        # drop output side graph state we no longer need\n"
                        "        for stage_output in stage_outputs_or_loss:\n"
                        "            if isinstance(stage_output, torch.Tensor):\n"
                        "                stage_output.detach_()\n"
                        "\n",
                        "                    inp.grad += dinput\n"
                        "\n"
                        "        # Drop output-side graph state we no longer need. A view "
                        "cannot be\n"
                        "        # detached in place (e.g. the output of an identity "
                        "autograd.Function,\n"
                        "        # such as an activation-checkpoint boundary); its graph "
                        "is freed when\n"
                        "        # the last reference to the output goes away, which for "
                        "outputs the\n"
                        "        # stage retains is after the weight backward.\n"
                        "        for stage_output in stage_outputs_or_loss:\n"
                        "            if isinstance(stage_output, torch.Tensor) and not "
                        "stage_output._is_view():\n"
                        "                stage_output.detach_()\n"
                        "\n",
                    )
                ],
            }
        ],
    },
]
