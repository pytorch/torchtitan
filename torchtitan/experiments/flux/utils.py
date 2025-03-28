import math

import torch
from einops import rearrange

from PIL import ExifTags, Image

from torch import Tensor

from torchtitan.experiments.flux.model import FluxModel

from torchtitan.experiments.flux.modules.HFEmbedder import HFEmbedder


# CONSTANTS FOR FLUX PREPROCESSING
PATCH_HEIGHT, PATCH_WIDTH = 2, 2
POSITION_DIM = 3
LATENT_CHANNELS = 16
IMG_LATENT_SIZE_RATIO = 8


def save_image(
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
) -> int:
    fn = output_name.format(idx=idx)
    print(f"Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    exif_data[ExifTags.Base.Model] = name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = prompt
    img.save(fn, exif=exif_data, quality=95, subsampling=0)
    idx += 1

    return idx


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def generate_noise_latent(
    num_samples: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    """Generate noise latents for the Flux flow model.

    Args:
        num_samples (int): Equal to batch_size.
        height (int): The height of the image.
        width (int): The width of the image.
        device (str | torch.device): The device to use.
        dtype (torch.dtype): The dtype to use.
        seed (int): The seed to use for randomize.

    Returns:
        Tensor: The noise latents.
            Shape: [num_samples, LATENT_CHANNELS, height // IMG_LATENT_SIZE_RATIO, width // IMG_LATENT_SIZE_RATIO]

    """

    return torch.randn(
        num_samples,
        LATENT_CHANNELS,
        height // IMG_LATENT_SIZE_RATIO,
        width // IMG_LATENT_SIZE_RATIO,
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)


def create_position_encoding_for_latents(
    bsz: int, latent_height: int, latent_width: int
) -> Tensor:
    """
    Create the packed latents' position encodings for the Flux flow model.

    Args:
        bsz (int): The batch size.
        latent_height (int): The height of the latent.
        latent_width (int): The width of the latent.

    Returns:
        Tensor: The position encodings.
            Shape: [bsz, (latent_height // PATCH_HEIGHT) * (latent_width // PATCH_WIDTH), POSITION_DIM)
    """
    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH

    position_encoding = torch.zeros(height, width, POSITION_DIM)

    row_indices = torch.arange(height)
    position_encoding[:, :, 1] = row_indices.unsqueeze(1)

    col_indices = torch.arange(width)
    position_encoding[:, :, 2] = col_indices.unsqueeze(0)

    # Flatten and repeat for the full batch
    # [height, width, 3] -> [bsz, height * width, 3]
    position_encoding = position_encoding.view(1, height * width, POSITION_DIM)
    position_encoding = position_encoding.repeat(bsz, 1, 1)

    return position_encoding


def pack_latents(x: Tensor) -> Tensor:
    """
    Rearrange latents from an image-like format into a sequence of patches.
    Equivalent to `einops.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`.

    Args:
        x (Tensor): The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]

    Returns:
        Tensor: The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
    """
    b, c, latent_height, latent_width = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH

    # [b, c, h*ph, w*ph] -> [b, c, h, w, ph, pw]
    x = x.unfold(2, PATCH_HEIGHT, PATCH_HEIGHT).unfold(3, PATCH_WIDTH, PATCH_WIDTH)

    # [b, c, h, w, ph, PW] -> [b, h, w, c, ph, PW]
    x = x.permute(0, 2, 3, 1, 4, 5)

    # [b, h, w, c, ph, PW] -> [b, h*w, c*ph*PW]
    return x.reshape(b, h * w, c * PATCH_HEIGHT * PATCH_WIDTH)


def denoise(
    model: FluxModel,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    target: Tensor,  # TODO(jianiw): This is place holder now
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens
    img_cond: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack_latent(x: Tensor, latent_height: int, latent_width: int) -> Tensor:
    """
    Rearrange latents from a sequence of patches into an image-like format.

    Args:
        x (Tensor): The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
        latent_height (int): The height of the unpacked latents.
        latent_width (int): The width of the unpacked latents.

    Returns:
        Tensor: The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]
    """
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(latent_height / 16),
        w=math.ceil(latent_width / 16),
        ph=2,
        pw=2,
    )
