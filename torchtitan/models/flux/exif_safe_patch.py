"""
EXIF-safe monkey patches for PIL, inspired by HF Datasets PR #7638.

Goal: prevent training crashes from broken EXIF metadata by wrapping:
- PIL.Image.Image.getexif()
- PIL.ImageOps.exif_transpose()

If EXIF parsing fails, we fall back to returning the original image / empty EXIF.

Additionally:
  - Enable PIL.ImageFile.LOAD_TRUNCATED_IMAGES to prevent crashes from
    `OSError: image file is truncated` during streaming/iterable decoding.
"""

from typing import Any


def apply() -> None:
    """Apply PIL safety patches (idempotent)."""

    try:
        import PIL.Image  # type: ignore
        import PIL.ImageFile  # type: ignore
        import PIL.ImageOps  # type: ignore
    except Exception:
        return

    # ---------------------------------
    # Allow truncated images to load
    # ---------------------------------
    # HF Datasets calls image.load() to avoid FD leaks; truncated images can raise.
    # Setting this flag makes PIL attempt to load what it can instead of crashing.
    try:
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # -----------------------------
    # Patch PIL.Image.Image.getexif
    # -----------------------------
    ImageCls = getattr(PIL.Image, "Image", None)
    if ImageCls is not None:
        orig_getexif = getattr(ImageCls, "getexif", None)
        if callable(orig_getexif) and not getattr(orig_getexif, "_tt_exif_safe", False):

            def _safe_getexif(self: Any) -> Any:
                try:
                    return orig_getexif(self)
                except Exception:
                    # Return an "empty" Exif object if possible; otherwise empty dict.
                    try:
                        exif_cls = getattr(PIL.Image, "Exif", None)
                        if exif_cls is not None:
                            return exif_cls()
                    except Exception:
                        pass
                    return {}

            setattr(_safe_getexif, "_tt_exif_safe", True)
            ImageCls.getexif = _safe_getexif  # type: ignore[assignment]

    # ---------------------------------
    # Patch PIL.ImageOps.exif_transpose
    # ---------------------------------
    orig_exif_transpose = getattr(PIL.ImageOps, "exif_transpose", None)
    if callable(orig_exif_transpose) and not getattr(
        orig_exif_transpose, "_tt_exif_safe", False
    ):

        def _safe_exif_transpose(image: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return orig_exif_transpose(image, *args, **kwargs)
            except Exception:
                # If EXIF transpose fails, return the image unchanged.
                return image

        setattr(_safe_exif_transpose, "_tt_exif_safe", True)
        PIL.ImageOps.exif_transpose = _safe_exif_transpose  # type: ignore[assignment]
