"""Image loading, saving and colour-space plumbing.

Everything inside photolift travels as float32 RGB in [0, 1], sRGB-encoded.
Conversions to/from linear light, LAB and luma live here so that the operator
modules never have to think about it.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

__all__ = [
    "load",
    "save",
    "to_float",
    "to_uint8",
    "to_uint16",
    "srgb_to_linear",
    "linear_to_srgb",
    "rgb_to_lab",
    "lab_to_rgb",
    "luma",
    "replace_luma",
]

_EXT_16BIT = {".png", ".tif", ".tiff"}


def load(path: str | os.PathLike) -> np.ndarray:
    """Read an image from disk as float32 RGB in [0, 1].

    Honours EXIF orientation, drops alpha, and promotes greyscale to 3 channels.
    Raises ``FileNotFoundError`` / ``ValueError`` rather than returning ``None``
    the way ``cv2.imread`` does.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.fromfile(path, dtype=np.uint8)  # unicode-safe on every platform
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"could not decode image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    out = to_float(img)
    return _apply_exif_orientation(out, path)


def save(
    img: np.ndarray,
    path: str | os.PathLike,
    quality: int = 95,
    bit_depth: int | None = None,
) -> str:
    """Write a float RGB image to disk. Returns the path written."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower() or ".png"
    if bit_depth is None:
        bit_depth = 16 if ext in _EXT_16BIT else 8

    arr = to_uint16(img) if bit_depth == 16 else to_uint8(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        if bit_depth == 16:  # JPEG cannot hold 16 bits
            bgr = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"could not encode image for {path}")
    buf.tofile(path)
    return path


# --------------------------------------------------------------------------
# dtype conversion
# --------------------------------------------------------------------------

def to_float(img: np.ndarray) -> np.ndarray:
    """Any dtype -> float32 in [0, 1]."""
    if img.dtype == np.float32:
        return np.clip(img, 0.0, 1.0)
    if img.dtype == np.float64:
        return np.clip(img.astype(np.float32), 0.0, 1.0)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    raise TypeError(f"unsupported dtype {img.dtype}")


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(to_float(img) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return img
    return np.clip(to_float(img) * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


# --------------------------------------------------------------------------
# colour spaces
# --------------------------------------------------------------------------

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0,1] -> linear light. Use before anything physical
    (blending, resampling energy, exposure scaling)."""
    a = 0.055
    lo = img / 12.92
    hi = np.power((np.clip(img, 0.0, None) + a) / (1 + a), 2.4)
    return np.where(img <= 0.04045, lo, hi).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    a = 0.055
    img = np.clip(img, 0.0, None)
    lo = img * 12.92
    hi = (1 + a) * np.power(img, 1 / 2.4) - a
    return np.clip(np.where(img <= 0.0031308, lo, hi), 0.0, 1.0).astype(np.float32)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """float RGB [0,1] -> CIE LAB with L in [0,100], a/b roughly [-128,127]."""
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0.0, 1.0)


def luma(img: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an sRGB-encoded image, single channel float32."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return (
        0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    ).astype(np.float32)


def replace_luma(img: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    """Swap an image's luminance while keeping hue and saturation.

    Implemented as a per-pixel scale of the RGB triple. Because :func:`luma` is
    a linear combination of those channels, scaling by ``new_y / old_y`` makes
    ``luma(result) == new_y`` exactly, and because all three channels move
    together the RGB ratios -- hence hue and HSV saturation -- are untouched.

    (Do not be tempted to write ``new_y`` into a CIE L* channel instead: L* is
    perceptual lightness on a 0-100 scale, not Rec.709 luma, and the two are
    not interchangeable. Doing so darkens midtones by roughly 20%.)
    """
    new_y = np.clip(new_y, 0.0, 1.0)
    old_y = luma(img)
    eps = 1e-4
    scale = (new_y + eps) / (old_y + eps)
    out = img * scale[..., None]

    # Near-black pixels have no ratio worth trusting; shift them additively.
    dark = old_y < 2e-3
    if np.any(dark):
        out[dark] = img[dark] + (new_y - old_y)[dark][:, None]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------

def _apply_exif_orientation(img: np.ndarray, path: str) -> np.ndarray:
    orientation = _read_exif_orientation(path)
    if orientation in (None, 1):
        return img
    if orientation == 2:
        return img[:, ::-1]
    if orientation == 3:
        return img[::-1, ::-1]
    if orientation == 4:
        return img[::-1]
    if orientation == 5:
        return np.transpose(img, (1, 0, 2))
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return np.transpose(img, (1, 0, 2))[::-1, ::-1]
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _read_exif_orientation(path: str) -> int | None:
    try:
        from PIL import ExifTags, Image  # noqa: PLC0415  (optional dependency)
    except ImportError:  # pragma: no cover
        return None
    try:
        with Image.open(path) as im:
            exif: Any = im.getexif()
            if not exif:
                return None
            for tag, name in ExifTags.TAGS.items():
                if name == "Orientation":
                    return exif.get(tag)
    except Exception:  # pragma: no cover - EXIF is never worth failing over
        return None
    return None
