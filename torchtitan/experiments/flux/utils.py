import math
from typing import Optional

import torch
from einops import rearrange

from PIL import ExifTags, Image

from torch import Tensor

from torchtitan.experiments.flux.model.model import FluxModel
from torchtitan.experiments.flux.model.modules.autoencoder import AutoEncoder
from torchtitan.experiments.flux.model.modules.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.sampling import get_schedule


# CONSTANTS FOR FLUX PREPROCESSING
PATCH_HEIGHT, PATCH_WIDTH = 2, 2
POSITION_DIM = 3
LATENT_CHANNELS = 16
IMG_LATENT_SIZE_RATIO = 8


def save_image(
    name: str,
    output_name: str,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
):
    print(f"Saving {output_name}")
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
    img.save(output_name, exif=exif_data, quality=95, subsampling=0)


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def preprocess_data(
    # arguments from the recipe
    device: torch.device,
    dtype: torch.dtype,
    *,
    # arguments from the config
    autoencoder: Optional[AutoEncoder],
    clip_encoder: FluxEmbedder,
    t5_encoder: FluxEmbedder,
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    """
    Take a batch of inputs and
    Args:
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
        autoencoer
        clip_encoder
        t5_encoder
        batch (dict[str, Tensor]): batch of data to preprocess
    """

    # The input of encoder should be torch.int type
    clip_tokens = batch["clip_tokens"].to(device=device, dtype=torch.int)
    t5_tokens = batch["t5_tokens"].to(device=device, dtype=torch.int)

    clip_text_encodings = clip_encoder(clip_tokens)
    t5_text_encodings = t5_encoder(t5_tokens)

    if autoencoder is not None:
        images = batch["image"].to(device=device, dtype=dtype)
        img_encodings = autoencoder.encode(images)
        batch["img_encodings"] = img_encodings

    batch["clip_encodings"] = clip_text_encodings.to(dtype)
    batch["t5_encodings"] = t5_text_encodings.to(dtype)

    return batch


def generate_noise_latent(
    bsz: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    """Generate noise latents for the Flux flow model.

    Args:
        bsz (int): batch_size.
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
        bsz,
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


def unpack_latents(x: Tensor, latent_height: int, latent_width: int) -> Tensor:
    """
    Rearrange latents from a sequence of patches into an image-like format.
    Equivalent to `einops.rearrange("b (h w) (c ph pw) -> b c (h ph) (w pw)")`.

    Args:
        x (Tensor): The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
        latent_height (int): The height of the unpacked latents.
        latent_width (int): The width of the unpacked latents.

    Returns:
        Tensor: The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]
    """
    b, _, c_ph_pw = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH
    c = c_ph_pw // (PATCH_HEIGHT * PATCH_WIDTH)

    # [b, h*w, c*ph*pw] -> [b, h, w, c, ph, pw]
    x = x.reshape(b, h, w, c, PATCH_HEIGHT, PATCH_WIDTH)

    # [b, h, w, c, ph, pw] -> [b, c, h, ph, w, pw]
    x = x.permute(0, 3, 1, 4, 2, 5)

    # [b, c, h, ph, w, pw] -> [b, c, h*ph, w*pw]
    return x.reshape(b, c, h * PATCH_HEIGHT, w * PATCH_WIDTH)


def generate_images(
    device: torch.device,
    dtype: torch.dtype,
    model: FluxModel,
    decoder: AutoEncoder,
    # image params:
    img_width: int,
    img_height: int,
    # sampling params:
    denoising_steps: int,
    seed: int,
    clip_encodings: Tensor,
    t5_encodings: Tensor,
    guidance: float = 4.0,
):

    bsz = clip_encodings.shape[0]
    latents = generate_noise_latent(bsz, img_height, img_width, device, dtype, seed)
    _, latent_channels, latent_height, latent_width = latents.shape

    # create denoising schedule
    timesteps = get_schedule(denoising_steps, latent_channels, shift=True)

    # create positional encodings
    latent_pos_enc = create_position_encoding_for_latents(
        bsz, latent_height, latent_width
    ).to(latents)
    text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM).to(latents)

    # convert img-like latents into sequences of patches
    latents = pack_latents(latents)

    # this is ignored for schnell
    guidance_vec = torch.full((bsz,), guidance, device=device, dtype=dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((bsz,), t_curr, dtype=dtype, device=device)
        pred = model(
            img=latents,
            img_ids=latent_pos_enc,
            txt=t5_encodings,
            txt_ids=text_pos_enc,
            y=clip_encodings,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        latents = latents + (t_prev - t_curr) * pred

    # convert sequences of patches into img-like latents
    latents = unpack_latents(latents, latent_height, latent_width)

    img = decoder.decode(latents)
    return img


def predict_noise(
    model: FluxModel,
    latents: Tensor,
    clip_encodings: Tensor,
    t5_encodings: Tensor,
    timesteps: Tensor,
    guidance: Optional[Tensor] = None,
) -> Tensor:
    """
    Use Flux's flow-matching model to predict the noise in image latents.
    Args:
        model (FluxFlowModel): The Flux flow model.
        latents (Tensor): Image encodings from the Flux autoencoder.
            Shape: [bsz, 16, latent height, latent width]
        clip_encodings (Tensor): CLIP text encodings.
            Shape: [bsz, 768]
        t5_encodings (Tensor): T5 text encodings.
            Shape: [bsz, sequence length, 256 or 512]
        timesteps (Tensor): The amount of noise (0 to 1).
            Shape: [bsz]
        guidance (Optional[Tensor]): The guidance value (1.5 to 4) if guidance-enabled model.
            Shape: [bsz]
            Default: None
        model_ctx (ContextManager): Optional context to wrap the model call (e.g. for activation offloading)
            Default: nullcontext
    Returns:
        Tensor: The noise prediction.
            Shape: [bsz, 16, latent height, latent width]
    """
    bsz, _, latent_height, latent_width = latents.shape

    with torch.no_grad():
        # Create positional encodings
        latent_pos_enc = create_position_encoding_for_latents(
            bsz, latent_height, latent_width
        )
        text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

        # Convert latent into a sequence of patches
        latents = pack_latents(latents)

    # Predict noise
    latent_noise_pred = model(
        img=latents,
        img_ids=latent_pos_enc.to(latents),
        txt=t5_encodings.to(latents),
        txt_ids=text_pos_enc.to(latents),
        y=clip_encodings.to(latents),
        timesteps=timesteps.to(latents),
        guidance=guidance.to(latents) if guidance is not None else None,
    )

    # Convert sequence of patches to latent shape
    latent_noise_pred = unpack_latents(latent_noise_pred, latent_height, latent_width)

    return latent_noise_pred
