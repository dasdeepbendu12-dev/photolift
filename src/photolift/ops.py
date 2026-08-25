"""The individual photographic operators.

Each function takes and returns a float32 RGB image in [0, 1] and is safe to
call on its own -- the pipeline just picks an order and a set of strengths.

Design rules kept throughout:
  * tone and detail work happens on luminance only, so nothing shifts hue;
  * anything physical (exposure, blending, resampling) happens in linear light;
  * every operator degrades gracefully to a no-op at strength 0.
"""

from __future__ import annotations

import cv2
import numpy as np

from .imageio import (
    lab_to_rgb,
    linear_to_srgb,
    luma,
    replace_luma,
    rgb_to_lab,
    srgb_to_linear,
)

__all__ = [
    "white_balance",
    "auto_exposure",
    "shadows_highlights",
    "local_contrast",
    "levels",
    "dehaze",
    "denoise",
    "flat_denoise",
    "deconvolve",
    "unsharp_mask",
    "clarity",
    "vibrance",
    "tone_curve",
    "add_grain",
]


# --------------------------------------------------------------------------
# colour / lighting
# --------------------------------------------------------------------------

def white_balance(
    img: np.ndarray,
    strength: float = 1.0,
    gains: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Neutralise a colour cast.

    ``gains`` comes from :func:`photolift.analysis.analyze`; pass your own to
    force a specific illuminant. ``strength`` blends between original and fully
    corrected, which matters for photos where the cast *is* the mood (sunsets,
    candlelight) and you only want to take the edge off.
    """
    if strength <= 0:
        return img
    if gains is None:
        from .analysis import _white_balance_gains  # local import: avoids cycle
        gains, _ = _white_balance_gains(img)

    g = np.asarray(gains, dtype=np.float32)
    g = 1.0 + (g - 1.0) * float(strength)

    lin = srgb_to_linear(img) * g.reshape(1, 1, 3)
    return linear_to_srgb(lin)


def auto_exposure(
    img: np.ndarray,
    target: float = 0.48,
    strength: float = 1.0,
    max_stops: float = 2.5,
) -> np.ndarray:
    """Pull the image's median luminance toward ``target``.

    Applied as a real exposure change in linear light (a multiply), then rolled
    off with a soft shoulder so that lifting a dark frame does not simply blow
    every highlight to paper white.
    """
    if strength <= 0:
        return img
    lin = srgb_to_linear(img)
    y = luma(lin)
    median = float(np.median(y))
    if median <= 1e-5:
        return img

    target_lin = float(srgb_to_linear(np.array([[[target]]], np.float32))[0, 0, 0])
    gain = target_lin / median
    stops = float(np.clip(np.log2(max(gain, 1e-6)), -max_stops, max_stops))
    gain = 2.0 ** (stops * float(strength))
    if abs(gain - 1.0) < 1e-3:
        return img

    lin = lin * gain
    if gain > 1.0:
        lin = _soft_shoulder(lin, knee=0.75)
    return linear_to_srgb(lin)


def _soft_shoulder(lin: np.ndarray, knee: float = 0.75) -> np.ndarray:
    """Compress everything above ``knee`` asymptotically toward 1.0."""
    over = lin > knee
    if not np.any(over):
        return lin
    out = lin.copy()
    x = (lin[over] - knee) / max(1.0 - knee, 1e-6)
    out[over] = knee + (1.0 - knee) * (x / (1.0 + x))
    return out


def shadows_highlights(
    img: np.ndarray,
    shadows: float = 0.0,
    highlights: float = 0.0,
    radius: float | None = None,
) -> np.ndarray:
    """Lift shadows and recover highlights independently.

    The masks come from a heavily blurred luminance channel, so the correction
    follows large tonal regions instead of individual pixels -- that is what
    keeps it from looking like a flat gamma lift. Edge-aware filtering on the
    mask suppresses the halos this technique is notorious for.
    """
    if abs(shadows) < 1e-3 and abs(highlights) < 1e-3:
        return img

    y = luma(img)
    h, w = y.shape
    if radius is None:
        radius = max(h, w) / 16.0
    base = _edge_aware_blur(y, radius)

    out = y.copy()
    if abs(shadows) > 1e-3:
        mask = np.clip(1.0 - base * 2.0, 0.0, 1.0) ** 2
        out = out + shadows * mask * (1.0 - out) * 0.9
    if abs(highlights) > 1e-3:
        mask = np.clip((base - 0.5) * 2.0, 0.0, 1.0) ** 2
        out = out - highlights * mask * out * 0.9

    return replace_luma(img, np.clip(out, 0.0, 1.0))


def _edge_aware_blur(y: np.ndarray, radius: float) -> np.ndarray:
    """Guided-filter style smoothing: big blur that still respects edges."""
    r = max(2, int(radius))
    eps = 0.01 ** 2
    ksize = (r * 2 + 1, r * 2 + 1)
    mean_i = cv2.blur(y, ksize)
    mean_ii = cv2.blur(y * y, ksize)
    var = np.maximum(mean_ii - mean_i * mean_i, 0.0)
    a = var / (var + eps)
    b = mean_i - a * mean_i
    a = cv2.blur(a, ksize)
    b = cv2.blur(b, ksize)
    return np.clip(a * y + b, 0.0, 1.0)


def local_contrast(
    img: np.ndarray,
    clip_limit: float = 1.6,
    tile_grid: int = 8,
    blend: float = 1.0,
) -> np.ndarray:
    """CLAHE on the L channel: adds contrast where the image is locally flat
    without wrecking the global tone. ``blend`` mixes it back with the input,
    which is the honest way to get "a bit of CLAHE" instead of the crunchy look."""
    if clip_limit <= 0 or blend <= 0:
        return img
    lab = rgb_to_lab(img)
    l8 = np.clip(lab[..., 0] / 100.0 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    lab[..., 0] = clahe.apply(l8).astype(np.float32) / 255.0 * 100.0
    out = lab_to_rgb(lab)
    return _blend(img, out, blend)


def levels(
    img: np.ndarray,
    low_pct: float = 0.2,
    high_pct: float = 99.8,
    strength: float = 1.0,
    preserve_clipping: bool = True,
) -> np.ndarray:
    """Percentile black/white point stretch on luminance.

    ``preserve_clipping`` refuses to stretch further into tones that are already
    clipped, so an image that is genuinely full-range is left alone.
    """
    if strength <= 0:
        return img
    y = luma(img)
    lo, hi = np.percentile(y, [low_pct, high_pct])
    if hi - lo < 1e-4:
        return img
    if preserve_clipping:
        lo = max(float(lo), 0.0)
        hi = min(float(hi), 1.0)

    stretched = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    new_y = y + (stretched - y) * float(strength)
    return replace_luma(img, new_y)


def dehaze(img: np.ndarray, strength: float = 0.5, patch: int = 15) -> np.ndarray:
    """Dark-channel-prior haze removal.

    Recovers contrast and colour in flat, milky images (distant landscapes,
    photos through glass, scans). ``strength`` scales the transmission estimate;
    above ~0.8 it starts to look synthetic.
    """
    if strength <= 0:
        return img
    lin = srgb_to_linear(img)

    dark = cv2.erode(np.min(lin, axis=2),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))
    # Atmospheric light: brightest 0.1% of the dark channel.
    n = dark.size
    k = max(1, int(n * 0.001))
    idx = np.argpartition(dark.ravel(), -k)[-k:]
    a = np.maximum(lin.reshape(-1, 3)[idx].max(axis=0), 1e-3)

    omega = float(np.clip(strength, 0.0, 1.0)) * 0.95
    norm = np.min(lin / a.reshape(1, 1, 3), axis=2)
    t = 1.0 - omega * cv2.erode(
        norm, cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    )
    t = _edge_aware_blur(np.clip(t, 0.0, 1.0), max(img.shape[:2]) / 60.0)
    # The transmission floor is a noise control, not a detail. Dividing by t
    # multiplies everything in that region -- including grain -- by 1/t, so a
    # floor of 0.1 is a 10x noise gain in exactly the dark areas where noise is
    # worst. 0.3 costs a little haze removal and buys clean shadows.
    t = np.clip(t, 0.30, 1.0)[..., None]

    out = (lin - a.reshape(1, 1, 3)) / t + a.reshape(1, 1, 3)
    return linear_to_srgb(np.clip(out, 0.0, 1.0))


# --------------------------------------------------------------------------
# detail
# --------------------------------------------------------------------------

def denoise(
    img: np.ndarray,
    strength: float = 0.5,
    chroma_strength: float | None = None,
    preserve_detail: float | None = None,
) -> np.ndarray:
    """Non-local-means denoise, luma and chroma tuned separately.

    Chroma noise is the ugly part of a high-ISO frame and can be crushed hard;
    luma noise reads as grain and should be treated more gently, because
    killing it also kills texture.

    ``preserve_detail`` blends part of the removed residual back in so surfaces
    keep their microstructure instead of turning to plastic. Left at ``None``
    it is derived from ``strength``, and that relationship is not optional: the
    residual contains the noise *and* the texture, so at high strength -- the
    case where there was a lot of noise -- feeding a fixed fraction back simply
    re-injects what was just removed. It has to taper to near zero exactly when
    denoising is working hardest.
    """
    if strength <= 0:
        return img
    if chroma_strength is None:
        chroma_strength = min(1.0, strength * 2.0)
    if preserve_detail is None:
        preserve_detail = float(np.clip(0.15 + 0.40 * (1.0 - strength), 0.15, 0.55))

    src = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # Deliberately asymmetric. Chroma noise has no redeeming quality, so it
    # gets hit hard; luma noise reads as grain, and pushing non-local means too
    # far on the luma channel produces its signature blotchy patches.
    h_luma = float(np.clip(strength, 0, 1) * 9.0)
    h_chroma = float(np.clip(chroma_strength, 0, 1) * 24.0)
    out = cv2.fastNlMeansDenoisingColored(src, None, h_luma, h_chroma, 7, 21)
    out = out.astype(np.float32) / 255.0

    if preserve_detail > 0:
        # Residual = what denoising removed. Its low-amplitude part is texture,
        # its high-amplitude part near edges is noise; feed back the former.
        residual = luma(img) - luma(out)
        texture = residual - cv2.GaussianBlur(residual, (0, 0), 1.2)
        new_y = np.clip(luma(out) + texture * float(preserve_detail), 0.0, 1.0)
        out = replace_luma(out, new_y)
    return out


def flat_denoise(img: np.ndarray, strength: float = 0.5, radius: float = 1.2,
                 detail_floor: float | None = None) -> np.ndarray:
    """Smooth only the *flat* regions -- skies, walls, skin, out-of-focus areas.

    This exists because of where it sits in the pipeline. Denoising happens
    early, before contrast is stretched; levels, CLAHE and sharpening then
    amplify whatever grain survived, and they amplify it most visibly in
    exactly the smooth areas where the eye has no detail to look at instead.
    Denoising harder up front is the wrong answer -- it costs texture
    everywhere to fix a problem that only shows up in flat areas.

    So: measure local variance on the *output*, build a mask of regions with
    no real detail, and smooth only there. Textured areas are untouched, which
    means this cleans the visible grain without costing any sharpness that
    matters.
    """
    if strength <= 0:
        return img

    y = luma(img)
    # Local standard deviation, cheaply, at a radius small enough to separate
    # "smooth wall" from "fine texture".
    k = (7, 7)
    mean = cv2.blur(y, k)
    var = np.maximum(cv2.blur(y * y, k) - mean * mean, 0.0)
    local_sd = np.sqrt(var)

    if detail_floor is None:
        # The threshold has to be set *relative to the noise*, not as an
        # absolute. Local standard deviation in a flat region is dominated by
        # the very noise being removed, so a fixed floor at, say, 0.02 declares
        # every 0.02-sigma noisy region "detailed" and the mask collapses to
        # zero -- the operator silently does nothing exactly when it is needed.
        # Measuring the noise and setting the floor a few sigma above it means
        # "flat" reliably means "no signal above the noise".
        from .analysis import _estimate_noise  # local import: avoids a cycle
        detail_floor = max(_estimate_noise(y) * 3.0, 0.006)

    # 1 where the region is featureless, falling to 0 once real detail appears.
    flat = np.clip(1.0 - local_sd / max(detail_floor, 1e-4), 0.0, 1.0) ** 2
    flat = cv2.GaussianBlur(flat, (0, 0), 2.0) * float(np.clip(strength, 0, 1))

    smooth = cv2.bilateralFilter(img, 0, 0.06, radius * 4.0)
    return np.clip(img + (smooth - img) * flat[..., None], 0.0, 1.0)


def deconvolve(
    img: np.ndarray,
    sigma: float = 1.0,
    iterations: int = 12,
    damping: float = 0.02,
) -> np.ndarray:
    """Richardson-Lucy deconvolution against a Gaussian PSF.

    This is the only operator here that genuinely recovers focus rather than
    faking it: it inverts a blur model instead of boosting edge contrast. It is
    also the one that will ring if you overdrive it -- keep ``sigma`` near the
    measured ``blur_sigma`` and iterations modest. Luminance only.
    """
    if sigma <= 0.05 or iterations < 1:
        return img

    y = np.clip(luma(img), 1e-4, 1.0).astype(np.float32)
    psf = _gaussian_psf(sigma)
    psf_flip = psf[::-1, ::-1].copy()

    estimate = y.copy()
    for _ in range(int(iterations)):
        conv = cv2.filter2D(estimate, cv2.CV_32F, psf, borderType=cv2.BORDER_REFLECT)
        ratio = y / np.maximum(conv, damping)
        correction = cv2.filter2D(ratio, cv2.CV_32F, psf_flip,
                                  borderType=cv2.BORDER_REFLECT)
        estimate = np.clip(estimate * correction, 0.0, 1.0)

    # Ringing control: never let a pixel exceed the local min/max of the input
    # by much. Cheap, and kills the worst of the Gibbs overshoot.
    estimate = _clamp_to_local_range(estimate, y, radius=max(1, int(sigma * 2)))
    return replace_luma(img, estimate)


def _gaussian_psf(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(sigma * 3)))
    k = cv2.getGaussianKernel(radius * 2 + 1, sigma).astype(np.float32)
    psf = k @ k.T
    return psf / psf.sum()


def _clamp_to_local_range(est: np.ndarray, ref: np.ndarray, radius: int,
                          slack: float = 0.12) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1,) * 2)
    lo = cv2.erode(ref, kernel) - slack
    hi = cv2.dilate(ref, kernel) + slack
    return np.clip(est, lo, hi)


def unsharp_mask(
    img: np.ndarray,
    amount: float = 0.6,
    radius: float = 1.0,
    threshold: float = 0.01,
    halo_control: bool = True,
) -> np.ndarray:
    """Classic output sharpening on luminance, with an edge threshold so flat
    areas (skin, sky, sensor noise) are left alone, and optional halo control
    that clamps overshoot to the local tonal range."""
    if amount <= 0:
        return img
    y = luma(img)
    blurred = cv2.GaussianBlur(y, (0, 0), max(radius, 0.1))
    detail = y - blurred

    if threshold > 0:
        mask = np.clip((np.abs(detail) - threshold) / max(threshold, 1e-6), 0.0, 1.0)
        detail = detail * mask

    out = y + detail * float(amount)
    if halo_control:
        out = _clamp_to_local_range(out, y, radius=max(1, int(radius * 2)), slack=0.08)
    return replace_luma(img, np.clip(out, 0.0, 1.0))


def clarity(img: np.ndarray, amount: float = 0.3, radius: float | None = None
            ) -> np.ndarray:
    """Midtone local contrast ("punch") -- an unsharp mask with a large radius,
    weighted to spare shadows and highlights so it does not clip."""
    if amount <= 0:
        return img
    y = luma(img)
    if radius is None:
        radius = max(img.shape[:2]) / 100.0
    base = cv2.GaussianBlur(y, (0, 0), max(radius, 1.0))
    detail = y - base
    midtone = 1.0 - np.abs(y - 0.5) * 2.0  # 1 at mid grey, 0 at the extremes
    out = y + detail * float(amount) * np.clip(midtone, 0.0, 1.0)
    return replace_luma(img, np.clip(out, 0.0, 1.0))


# --------------------------------------------------------------------------
# colour finishing
# --------------------------------------------------------------------------

def vibrance(img: np.ndarray, amount: float = 0.2, protect_skin: bool = True
             ) -> np.ndarray:
    """Saturation that scales inversely with existing chroma.

    Muted colours get most of the boost, already-vivid ones barely move, so it
    does not turn every red into a solid block. ``protect_skin`` further damps
    the orange-red hue band where oversaturation is most obvious.
    """
    if abs(amount) < 1e-3:
        return img
    lab = rgb_to_lab(img)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)

    weight = 1.0 - np.clip(chroma / 60.0, 0.0, 1.0)
    if protect_skin:
        hue = np.arctan2(b, a)
        skin = np.exp(-((hue - 0.5) ** 2) / (2 * 0.35 ** 2))
        weight = weight * (1.0 - 0.6 * skin)

    scale = 1.0 + float(amount) * weight
    lab[..., 1] = a * scale
    lab[..., 2] = b * scale
    return lab_to_rgb(lab)


def tone_curve(img: np.ndarray, contrast: float = 0.0, pivot: float = 0.5
               ) -> np.ndarray:
    """Gentle S-curve contrast around ``pivot``, applied to luminance."""
    if abs(contrast) < 1e-3:
        return img
    y = luma(img)
    x = np.clip(y - pivot, -pivot, 1 - pivot)
    curved = pivot + x + float(contrast) * x * (1.0 - np.abs(x) / max(pivot, 1e-6)) * 0.5
    return replace_luma(img, np.clip(curved, 0.0, 1.0))


def add_grain(img: np.ndarray, amount: float = 0.0, size: float = 1.0,
              seed: int | None = None) -> np.ndarray:
    """Monochrome grain. Sounds backwards after denoising, but a trace of grain
    hides the plastic look of heavy noise reduction and upscaling, and it is
    what makes an enhanced image read as a photograph again."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, img.shape[:2]).astype(np.float32)
    if size > 1.0:
        noise = cv2.GaussianBlur(noise, (0, 0), size - 0.5)
        noise /= max(noise.std(), 1e-6)
    y = luma(img)
    # Grain is most visible in midtones on real film; mimic that.
    weight = 1.0 - np.abs(y - 0.5) * 1.4
    new_y = y + noise * float(amount) * np.clip(weight, 0.0, 1.0)
    return replace_luma(img, np.clip(new_y, 0.0, 1.0))


def _blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    if t >= 1.0:
        return b
    if t <= 0.0:
        return a
    return np.clip(a * (1 - t) + b * t, 0.0, 1.0)
