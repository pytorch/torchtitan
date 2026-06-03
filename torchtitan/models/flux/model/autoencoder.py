# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import torch
from einops import rearrange
from safetensors.torch import load_file as load_sft
from torch import nn, Tensor

from torchtitan.models.common.nn_modules import Conv2d, GroupNorm
from torchtitan.protocols.module import Module, ModuleList


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def _norm(num_channels: int) -> GroupNorm.Config:
    return GroupNorm.Config(num_groups=32, num_channels=num_channels, eps=1e-6)


class AttnBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.in_channels = config.in_channels

        self.norm = _norm(config.in_channels).build()

        conv1x1 = Conv2d.Config(
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            kernel_size=1,
        )
        self.q = conv1x1.build()
        self.k = conv1x1.build()
        self.v = conv1x1.build()
        self.proj_out = conv1x1.build()

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int
        out_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels

        self.norm1 = _norm(config.in_channels).build()
        self.conv1 = Conv2d.Config(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=3,
            padding=1,
        ).build()
        self.norm2 = _norm(config.out_channels).build()
        self.conv2 = Conv2d.Config(
            in_channels=config.out_channels,
            out_channels=config.out_channels,
            kernel_size=3,
            padding=1,
        ).build()
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2d.Config(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                kernel_size=1,
            ).build()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int

    def __init__(self, config: Config):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv2d.Config(
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            kernel_size=3,
            stride=2,
        ).build()

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.conv = Conv2d.Config(
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            kernel_size=3,
            padding=1,
        ).build()

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        resolution: int
        in_channels: int
        ch: int
        ch_mult: tuple[int, ...]
        num_res_blocks: int
        z_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        # downsampling
        self.conv_in = Conv2d.Config(
            in_channels=config.in_channels,
            out_channels=self.ch,
            kernel_size=3,
            padding=1,
        ).build()

        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = ModuleList()
            attn = ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock.Config(
                        in_channels=block_in, out_channels=block_out
                    ).build()
                )
                block_in = block_out
            down = Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample.Config(in_channels=block_in).build()
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = Module()
        self.mid.block_1 = ResnetBlock.Config(
            in_channels=block_in, out_channels=block_in
        ).build()
        self.mid.attn_1 = AttnBlock.Config(in_channels=block_in).build()
        self.mid.block_2 = ResnetBlock.Config(
            in_channels=block_in, out_channels=block_in
        ).build()

        # end
        self.norm_out = _norm(block_in).build()
        self.conv_out = Conv2d.Config(
            in_channels=block_in,
            out_channels=2 * config.z_channels,
            kernel_size=3,
            padding=1,
        ).build()

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # pyrefly: ignore[bad-index, not-callable]
                h = self.down[i_level].block[i_block](hs[-1])
                # pyrefly: ignore [bad-argument-type]
                if len(self.down[i_level].attn) > 0:
                    # pyrefly: ignore[bad-index, not-callable]
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # pyrefly: ignore [not-callable]
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        # pyrefly: ignore [not-callable]
        h = self.mid.block_1(h)
        # pyrefly: ignore [not-callable]
        h = self.mid.attn_1(h)
        # pyrefly: ignore [not-callable]
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ch: int
        out_ch: int
        ch_mult: tuple[int, ...]
        num_res_blocks: int
        in_channels: int
        resolution: int
        z_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = Conv2d.Config(
            in_channels=config.z_channels,
            out_channels=block_in,
            kernel_size=3,
            padding=1,
        ).build()

        # middle
        self.mid = Module()
        self.mid.block_1 = ResnetBlock.Config(
            in_channels=block_in, out_channels=block_in
        ).build()
        self.mid.attn_1 = AttnBlock.Config(in_channels=block_in).build()
        self.mid.block_2 = ResnetBlock.Config(
            in_channels=block_in, out_channels=block_in
        ).build()

        # upsampling
        self.up = ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = ModuleList()
            attn = ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock.Config(
                        in_channels=block_in, out_channels=block_out
                    ).build()
                )
                block_in = block_out
            up = Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample.Config(in_channels=block_in).build()
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = _norm(block_in).build()
        self.conv_out = Conv2d.Config(
            in_channels=block_in,
            out_channels=config.out_ch,
            kernel_size=3,
            padding=1,
        ).build()

    def forward(self, z: Tensor) -> Tensor:
        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        # pyrefly: ignore [not-callable]
        h = self.mid.block_1(h)
        # pyrefly: ignore [not-callable]
        h = self.mid.attn_1(h)
        # pyrefly: ignore [not-callable]
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # pyrefly: ignore[bad-index, not-callable]
                h = self.up[i_level].block[i_block](h)
                # pyrefly: ignore [bad-argument-type]
                if len(self.up[i_level].attn) > 0:
                    # pyrefly: ignore[bad-index, not-callable]
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # pyrefly: ignore [not-callable]
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        sample: bool = True
        chunk_dim: int = 1

    def __init__(self, config: Config):
        super().__init__()
        self.sample = config.sample
        self.chunk_dim = config.chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        resolution: int = 256
        in_channels: int = 3
        ch: int = 128
        out_ch: int = 3
        ch_mult: tuple[int, ...] = (1, 2, 4, 4)
        num_res_blocks: int = 2
        z_channels: int = 16
        scale_factor: float = 0.3611
        shift_factor: float = 0.1159

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = Encoder.Config(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        ).build()
        self.decoder = Decoder.Config(
            resolution=config.resolution,
            in_channels=config.in_channels,
            ch=config.ch,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            z_channels=config.z_channels,
        ).build()
        self.reg = DiagonalGaussian.Config().build()

        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def load_ae(
    ckpt_path: str,
    autoencoder_config: AutoEncoder.Config,
    device: str | torch.device = "cuda",
    dtype=torch.bfloat16,
    random_init=False,
) -> AutoEncoder:
    """Build the autoencoder from a Config and optionally load weights.

    Args:
        ckpt_path: Path to the autoencoder checkpoint.
        autoencoder_config: AutoEncoder.Config to instantiate.
        device: Device to load the autoencoder to.
        dtype: Target dtype after loading.
        random_init: If True, skip checkpoint loading.
    """
    with torch.device(device):
        ae = autoencoder_config.build()

    if random_init:
        return ae.to(dtype=dtype)

    if not os.path.exists(ckpt_path):
        raise ValueError(
            f"Autoencoder path {ckpt_path} does not exist. Please download it first."
        )

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if len(missing) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        if len(unexpected) > 0:
            print(
                f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected)
            )
    return ae.to(dtype=dtype)
