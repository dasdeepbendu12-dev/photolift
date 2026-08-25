"""Resolution increase.

Be clear-eyed about what each backend can do:

``lanczos``     interpolation. Adds pixels, adds zero information. Fast, safe.
``iterative``   Lanczos plus iterative back-projection -- each pass checks that
                downsampling the result reproduces the input and corrects the
                difference. Genuinely sharper than plain interpolation, still
                inventing nothing; this is the default.
``dnn``         OpenCV's ``dnn_superres`` (EDSR / ESPCN / FSRCNN / LapSRN).
                A learned model, so it *does* hallucinate plausible detail.
                Needs a ``.pb`` model file on disk.
``realesrgan``  Real-ESRGAN if the optional package is installed. Best quality
                on real degraded photos, heaviest dependency, and the most
                willing to invent faces and text that were never there.

Everything falls back down this list rather than raising, unless you ask for a
specific backend by name.
"""

from __future__ import annotations

import os
from typing import Callable

import cv2
import numpy as np

from .imageio import linear_to_srgb, srgb_to_linear

__all__ = ["upscale", "available_backends", "BACKENDS"]

BACKENDS = ("auto", "lanczos", "cubic", "iterative", "dnn", "realesrgan")

_DNN_MODEL_ENV = "PHOTOLIFT_SR_MODEL"
_DNN_ARCH_SCALES = {"edsr": (2, 3, 4), "espcn": (2, 3, 4),
                    "fsrcnn": (2, 3, 4), "lapsrn": (2, 4, 8)}


def upscale(
    img: np.ndarray,
    scale: float = 2.0,
    backend: str = "auto",
    model_path: str | None = None,
    tile: int | None = 512,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Return ``img`` enlarged by ``scale``.

    ``scale`` may be fractional; learned backends work at their native integer
    factor and the result is resampled to the exact requested size.
    ``tile`` bounds peak memory for the learned backends (``None`` disables).
    """
    if scale <= 1.0 + 1e-6:
        if scale < 1.0 - 1e-6:
            return _resize_linear(img, scale, cv2.INTER_AREA)
        return img

    backend = backend.lower()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")

    if backend == "auto":
        for candidate in ("realesrgan", "dnn", "iterative"):
            if _backend_ready(candidate, model_path, scale):
                backend = candidate
                break
        else:
            backend = "iterative"

    if progress:
        progress(f"upscale x{scale:g} via {backend}")

    if backend == "cubic":
        return _resize_linear(img, scale, cv2.INTER_CUBIC)
    if backend == "lanczos":
        return _resize_linear(img, scale, cv2.INTER_LANCZOS4)
    if backend == "iterative":
        return _iterative_backprojection(img, scale)
    if backend == "dnn":
        return _dnn_superres(img, scale, model_path, tile)
    if backend == "realesrgan":
        return _realesrgan(img, scale, tile)
    raise AssertionError("unreachable")


def available_backends(model_path: str | None = None) -> dict[str, bool]:
    """Which backends this machine can actually run right now."""
    return {b: _backend_ready(b, model_path, 2.0)
            for b in BACKENDS if b not in ("auto",)}


# --------------------------------------------------------------------------
# classical
# --------------------------------------------------------------------------

def _target_size(img: np.ndarray, scale: float) -> tuple[int, int]:
    h, w = img.shape[:2]
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _resize_linear(img: np.ndarray, scale: float, interp: int) -> np.ndarray:
    """Resample in linear light.

    Resampling sRGB values directly is the single most common way to lose
    energy at high-contrast edges; converting first costs little and keeps
    bright detail from darkening.
    """
    size = _target_size(img, scale)
    lin = srgb_to_linear(img)
    out = cv2.resize(lin, size, interpolation=interp)
    return linear_to_srgb(out)


def _iterative_backprojection(img: np.ndarray, scale: float,
                              iterations: int = 8, step: float = 0.9
                              ) -> np.ndarray:
    """Lanczos seed, then repeatedly enforce consistency with the input.

    At every pass the current estimate is downsampled back to the original
    resolution; whatever it fails to reproduce is upsampled and added back.
    Converges to an estimate that is both smooth and faithful, which reads as
    noticeably crisper than interpolation without any learned prior.
    """
    lr = srgb_to_linear(img)
    h, w = lr.shape[:2]
    size = _target_size(img, scale)
    est = cv2.resize(lr, size, interpolation=cv2.INTER_LANCZOS4)

    for _ in range(iterations):
        down = cv2.resize(est, (w, h), interpolation=cv2.INTER_AREA)
        error = lr - down
        # Stop early once the residual stops mattering.
        if float(np.abs(error).mean()) < 1e-5:
            break
        correction = cv2.resize(error, size, interpolation=cv2.INTER_LANCZOS4)
        est = np.clip(est + correction * step, 0.0, 1.0)

    return linear_to_srgb(est)


# --------------------------------------------------------------------------
# learned backends
# --------------------------------------------------------------------------

def _resolve_model(model_path: str | None) -> str | None:
    path = model_path or os.environ.get(_DNN_MODEL_ENV)
    return path if path and os.path.exists(path) else None


def _dnn_superres(img: np.ndarray, scale: float, model_path: str | None,
                  tile: int | None) -> np.ndarray:
    path = _resolve_model(model_path)
    if path is None:
        raise RuntimeError(
            "dnn backend needs a super-resolution model. Download one (e.g. "
            "EDSR_x2.pb from opencv/opencv_contrib's dnn_superres models) and "
            f"pass model_path=... or set ${_DNN_MODEL_ENV}."
        )
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
    except AttributeError as exc:  # pragma: no cover
        raise RuntimeError(
            "cv2.dnn_superres is missing -- install opencv-contrib-python"
        ) from exc

    arch, native = _parse_model_filename(path)
    sr.readModel(path)
    sr.setModel(arch, native)

    def run(chunk: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(np.clip(chunk * 255, 0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        out = sr.upsample(bgr)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    out = _tiled(img, run, native, tile)
    return _fit_exact(out, img, scale)


def _parse_model_filename(path: str) -> tuple[str, int]:
    name = os.path.basename(path).lower()
    arch = next((a for a in _DNN_ARCH_SCALES if a in name), None)
    if arch is None:
        raise ValueError(
            f"cannot infer architecture from {name!r}; expected the stock naming "
            "like EDSR_x2.pb / FSRCNN_x4.pb"
        )
    factor = next((f for f in _DNN_ARCH_SCALES[arch] if f"x{f}" in name), None)
    if factor is None:
        raise ValueError(f"cannot infer scale factor from {name!r}")
    return arch, factor


def _realesrgan(img: np.ndarray, scale: float, tile: int | None) -> np.ndarray:
    try:  # pragma: no cover - optional heavyweight dependency
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "realesrgan backend needs: pip install photolift[ai]"
        ) from exc

    native = 4  # the stock x4plus weights
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=native)
    upsampler = RealESRGANer(
        scale=native,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/"
                   "v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=tile or 0,
        tile_pad=10,
        half=False,
    )
    bgr = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out, _ = upsampler.enhance(bgr, outscale=native)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return _fit_exact(out, img, scale)


def _backend_ready(backend: str, model_path: str | None, scale: float) -> bool:
    if backend in ("lanczos", "cubic", "iterative"):
        return True
    if backend == "dnn":
        return _resolve_model(model_path) is not None and hasattr(cv2, "dnn_superres")
    if backend == "realesrgan":
        try:  # pragma: no cover
            import realesrgan  # noqa: F401
            return True
        except ImportError:
            return False
    return False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _fit_exact(out: np.ndarray, src: np.ndarray, scale: float) -> np.ndarray:
    """Learned models only work at their native factor; land on the exact size
    the caller asked for (downsampling from a larger result when possible,
    which is itself a quality win)."""
    want = _target_size(src, scale)
    if (out.shape[1], out.shape[0]) == want:
        return out
    interp = cv2.INTER_AREA if out.shape[1] > want[0] else cv2.INTER_LANCZOS4
    lin = cv2.resize(srgb_to_linear(out), want, interpolation=interp)
    return linear_to_srgb(lin)


def _tiled(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray],
           factor: int, tile: int | None, overlap: int = 16) -> np.ndarray:
    """Run ``fn`` over overlapping tiles and feather the seams.

    Without this, a 24MP input through a learned model is an out-of-memory
    error; with it, memory is bounded by tile size regardless of input.
    """
    h, w = img.shape[:2]
    if not tile or (h <= tile and w <= tile):
        return fn(img)

    out = np.zeros((h * factor, w * factor, 3), np.float32)
    weight = np.zeros((h * factor, w * factor, 1), np.float32)

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ys, xs = max(0, y0 - overlap), max(0, x0 - overlap)
            ye, xe = min(h, y0 + tile + overlap), min(w, x0 + tile + overlap)
            piece = fn(img[ys:ye, xs:xe])

            ph, pw = piece.shape[:2]
            mask = _feather(ph, pw, overlap * factor)
            oy, ox = ys * factor, xs * factor
            out[oy:oy + ph, ox:ox + pw] += piece * mask
            weight[oy:oy + ph, ox:ox + pw] += mask

    return np.clip(out / np.maximum(weight, 1e-6), 0.0, 1.0)


def _feather(h: int, w: int, pad: int) -> np.ndarray:
    pad = max(1, min(pad, h // 2, w // 2))
    ramp_y = np.ones(h, np.float32)
    ramp_x = np.ones(w, np.float32)
    ramp_y[:pad] = ramp_y[-pad:][::-1] = np.linspace(0.05, 1.0, pad, dtype=np.float32)
    ramp_x[:pad] = ramp_x[-pad:][::-1] = np.linspace(0.05, 1.0, pad, dtype=np.float32)
    return (ramp_y[:, None] * ramp_x[None, :])[..., None]
