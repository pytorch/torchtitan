"""Minimal ConvNeXt copy for path training."""

from __future__ import annotations

import re
from functools import partial

import torch
import torch.nn as nn
from timm.layers import (
    AvgPool2dSame,
    DropPath,
    Mlp,
    NormMlpClassifierHead,
    calculate_drop_path_rates,
    create_conv2d,
    get_act_layer,
    LayerNorm,
    LayerNorm2d,
    to_ntuple,
    trunc_normal_,
)
from timm.models import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq, named_apply


__all__ = [
    "CONVNEXT_FLAVORS",
    "ConvNeXt",
    "convnext_base",
    "convnext_small",
    "convnext_tiny",
    "convnext_xxlarge",
    "create_convnext",
    "pretrained_name",
]


CONVNEXT_FLAVORS = {
    "convnext_atto": {
        "depths": (2, 2, 6, 2),
        "dims": (40, 80, 160, 320),
        "pretrained": "convnext_atto.d2_in1k",
    },
    "convnext_femto": {
        "depths": (2, 2, 6, 2),
        "dims": (48, 96, 192, 384),
        "pretrained": "convnext_femto.d1_in1k",
    },
    "convnext_pico": {
        "depths": (2, 2, 6, 2),
        "dims": (64, 128, 256, 512),
        "pretrained": "convnext_pico.d1_in1k",
    },
    "convnext_tiny": {
        "depths": (3, 3, 9, 3),
        "dims": (96, 192, 384, 768),
        "pretrained": "convnext_tiny.in12k_ft_in1k",
    },
    "convnext_small": {
        "depths": (3, 3, 27, 3),
        "dims": (96, 192, 384, 768),
        "pretrained": "convnext_small.dinov3_lvd1689m",
    },
    "convnext_quarterxxl": {
        "depths": (3, 4, 30, 3),
        "dims": (96, 192, 384, 768),
        "pretrained": "convnext_small.in12k_ft_in1k",
    },
    "convnext_thirdxxl": {
        "depths": (3, 4, 30, 3),
        "dims": (128, 256, 512, 1024),
        "pretrained": "convnext_base.clip_laion2b_augreg_ft_in1k",
    },
    "convnext_base": {
        "depths": (3, 3, 27, 3),
        "dims": (128, 256, 512, 1024),
        "pretrained": "convnext_base.clip_laion2b_augreg_ft_in1k",
    },
    "convnext_xxlarge": {
        "depths": (3, 4, 30, 3),
        "dims": (384, 768, 1536, 3072),
        "pretrained": "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
    },
}


class Downsample(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        dilation: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = (
            create_conv2d(in_chs, out_chs, 1, stride=1, **dd)
            if in_chs != out_chs
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int | None = None,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int | tuple[int, int] = (1, 1),
        mlp_ratio: float = 4,
        conv_bias: bool = True,
        ls_init_value: float | None = 1e-6,
        act_layer="gelu",
        drop_path: float = 0.0,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        self.ls_init_value = ls_init_value
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
            **dd,
        )
        self.norm = LayerNorm(out_chs, eps=norm_eps, **dd)
        self.mlp = Mlp(out_chs, int(mlp_ratio * out_chs), act_layer=get_act_layer(act_layer), **dd)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs, **dd)) if ls_init_value is not None else None
        self.shortcut = (
            Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0], **dd)
            if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def reset_parameters(self) -> None:
        if self.gamma is not None:
            nn.init.constant_(self.gamma, self.ls_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_dw(x).permute(0, 2, 3, 1)
        x = self.mlp(self.norm(x)).permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return self.drop_path(x) + self.shortcut(shortcut)


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 7,
        stride: int = 2,
        depth: int = 2,
        dilation: tuple[int, int] = (1, 1),
        drop_path_rates: list[float] | None = None,
        ls_init_value: float = 1e-6,
        conv_bias: bool = True,
        act_layer="gelu",
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.grad_checkpointing = False
        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = "same" if dilation[1] > 1 else 0
            self.downsample = nn.Sequential(
                LayerNorm2d(in_chs, eps=norm_eps, **dd),
                create_conv2d(
                    in_chs,
                    out_chs,
                    ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                    **dd,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=(drop_path_rates or [0.0] * depth)[i],
                    ls_init_value=ls_init_value,
                    conv_bias=conv_bias,
                    act_layer=act_layer,
                    norm_eps=norm_eps,
                    **dd,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            return checkpoint_seq(self.blocks, x)
        return self.blocks(x)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        output_stride: int = 32,
        depths: tuple[int, ...] = (3, 4, 30, 3),
        dims: tuple[int, ...] = (384, 768, 1536, 3072),
        kernel_sizes: int | tuple[int, ...] = 7,
        ls_init_value: float | None = 1e-6,
        head_init_scale: float = 1.0,
        conv_bias: bool = True,
        act_layer="gelu",
        norm_eps: float = 1e-5,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert output_stride in (8, 16, 32)
        dd = {"device": device, "dtype": dtype}
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.head_init_scale = head_init_scale
        self.feature_info = []

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=conv_bias, **dd),
            LayerNorm2d(dims[0], eps=norm_eps, **dd),
        )
        self.stages = nn.Sequential()
        dp_rates = calculate_drop_path_rates(drop_path_rate, depths, stagewise=True)
        stages = []
        prev_chs = dims[0]
        curr_stride = 4
        dilation = 1
        for i in range(4):
            stride = 2 if i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    dims[i],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_bias=conv_bias,
                    act_layer=act_layer,
                    norm_eps=norm_eps,
                    **dd,
                )
            )
            prev_chs = dims[i]
            self.feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=f"stages.{i}"))
        self.stages = nn.Sequential(*stages)
        self.num_features = self.head_hidden_size = prev_chs
        self.norm_pre = nn.Identity()
        self.head = NormMlpClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            norm_layer=partial(LayerNorm2d, eps=norm_eps),
            act_layer="gelu",
            **dd,
        )
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        for stage in self.stages:
            stage.grad_checkpointing = enable

    def apply_activation_checkpointing(self, wrap, mode: str, base_fqn: str) -> None:
        if mode == "full":
            for stage_id, stage in enumerate(self.stages):
                for block_id, block in enumerate(stage.blocks):
                    stage.blocks[block_id] = wrap(
                        block,
                        f"{base_fqn}.stages.{stage_id}.blocks.{block_id}",
                    )
        else:
            for stage_id, stage in enumerate(self.stages):
                self.stages[stage_id] = wrap(stage, f"{base_fqn}.stages.{stage_id}")

    def apply_fsdp(
        self,
        shard,
        reshard_after_forward: bool,
        head_reshard_after_forward: bool,
    ) -> None:
        shard(self.stem, reshard_after_forward)
        for stage in self.stages:
            for block in getattr(stage, "blocks", ()):
                shard(block, reshard_after_forward)
            shard(stage, reshard_after_forward)
        shard(self.head, head_reshard_after_forward)

    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def init_path_weights(self) -> None:
        named_apply(partial(_init_weights, head_init_scale=self.head_init_scale), self)
        for module in self.modules():
            if isinstance(module, ConvNeXtBlock):
                module.reset_parameters()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.norm_pre(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def _init_weights(module: nn.Module, name: str | None = None, head_init_scale: float = 1.0) -> None:
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
        if name and "head." in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    if "head.norm.weight" in state_dict or "norm_pre.weight" in state_dict:
        return state_dict
    if "model" in state_dict:
        state_dict = state_dict["model"]

    out_dict = {}
    if "visual.trunk.stem.0.weight" in state_dict:
        out_dict = {k.replace("visual.trunk.", ""): v for k, v in state_dict.items() if k.startswith("visual.trunk.")}
        if "visual.head.proj.weight" in state_dict:
            out_dict["head.fc.weight"] = state_dict["visual.head.proj.weight"]
            out_dict["head.fc.bias"] = torch.zeros(state_dict["visual.head.proj.weight"].shape[0])
        elif "visual.head.mlp.fc1.weight" in state_dict:
            out_dict["head.pre_logits.fc.weight"] = state_dict["visual.head.mlp.fc1.weight"]
            out_dict["head.pre_logits.fc.bias"] = state_dict["visual.head.mlp.fc1.bias"]
            out_dict["head.fc.weight"] = state_dict["visual.head.mlp.fc2.weight"]
            out_dict["head.fc.bias"] = torch.zeros(state_dict["visual.head.mlp.fc2.weight"].shape[0])
        return out_dict

    for k, v in state_dict.items():
        k = k.replace("downsample_layers.0.", "stem.")
        k = re.sub(r"stages.([0-9]+).([0-9]+)", r"stages.\1.blocks.\2", k)
        k = re.sub(r"downsample_layers.([0-9]+).([0-9]+)", r"stages.\1.downsample.\2", k)
        k = k.replace("dwconv", "conv_dw")
        k = k.replace("pwconv", "mlp.fc")
        k = k.replace("head.", "head.fc.")
        if k.startswith("norm."):
            k = k.replace("norm", "head.norm")
        if v.ndim == 2 and "head" not in k:
            v = v.reshape(model.state_dict()[k].shape)
        out_dict[k] = v
    return out_dict

def _create_convnext(variant: str, pretrained: bool = False, **kwargs):
    return build_model_with_cfg(
        ConvNeXt,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )


def pretrained_name(flavor: str) -> str:
    return CONVNEXT_FLAVORS[flavor]["pretrained"]


def create_convnext(flavor: str, pretrained: bool = False, **kwargs) -> ConvNeXt:
    model_args = {
        "depths": CONVNEXT_FLAVORS[flavor]["depths"],
        "dims": CONVNEXT_FLAVORS[flavor]["dims"],
        "norm_eps": kwargs.pop("norm_eps", 1e-5),
    }
    return _create_convnext(flavor, pretrained=pretrained, **dict(model_args, **kwargs))


def convnext_tiny(pretrained: bool = False, **kwargs) -> ConvNeXt:
    return create_convnext("convnext_tiny", pretrained=pretrained, **kwargs)


def convnext_small(pretrained: bool = False, **kwargs) -> ConvNeXt:
    return create_convnext("convnext_small", pretrained=pretrained, **kwargs)


def convnext_base(pretrained: bool = False, **kwargs) -> ConvNeXt:
    return create_convnext("convnext_base", pretrained=pretrained, **kwargs)


def convnext_xxlarge(pretrained: bool = False, **kwargs) -> ConvNeXt:
    return create_convnext("convnext_xxlarge", pretrained=pretrained, **kwargs)
