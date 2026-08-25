"""Measure an image so the pipeline can decide how hard to push each knob.

Every metric here is normalised to a roughly interpretable range and is
resolution-aware where it matters, so that a 400px thumbnail and a 40MP frame
of the same scene score similarly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .imageio import luma, rgb_to_lab

__all__ = ["ImageStats", "analyze"]

# Longest edge used for the analysis pass. Metrics are computed on a downscaled
# copy so that cost is constant regardless of input size.
_ANALYSIS_LONG_EDGE = 768


@dataclass
class ImageStats:
    """A snapshot of the qualities photolift knows how to change."""

    width: int
    height: int
    megapixels: float

    # --- tone -------------------------------------------------------------
    brightness: float          # median luma, 0..1 (0.45-0.55 is well exposed)
    contrast: float            # std of luma, 0..~0.35
    dynamic_range: float       # p99 - p1 of luma, 0..1
    shadow_clip: float         # fraction of pixels crushed to black
    highlight_clip: float      # fraction of pixels blown to white
    entropy: float             # bits, 0..8; low means flat/hazy

    # --- detail -----------------------------------------------------------
    sharpness: float           # normalised Laplacian energy, 0..~1
    blur_sigma: float          # estimated Gaussian blur radius in px
                               # (under-reports on noisy frames -- see
                               # _estimate_blur_sigma)
    noise_sigma: float         # estimated noise sigma in [0,1] units

    # --- colour -----------------------------------------------------------
    saturation: float          # mean LAB chroma / 128, 0..1
    colorfulness: float        # Hasler-Susstrunk metric, 0..~1
    wb_cast: float             # magnitude of colour cast, 0..1
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # --- derived verdicts -------------------------------------------------
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"{self.width}x{self.height} ({self.megapixels:.1f}MP)",
            f"brightness {self.brightness:.2f}",
            f"contrast {self.contrast:.3f}",
            f"sharpness {self.sharpness:.3f} (blur ~{self.blur_sigma:.2f}px)",
            f"noise {self.noise_sigma * 255:.1f}/255",
            f"saturation {self.saturation:.2f}",
            f"cast {self.wb_cast:.3f}",
        ]
        line = " | ".join(parts)
        if self.flags:
            line += "\n  flags: " + ", ".join(self.flags)
        return line


def analyze(img: np.ndarray) -> ImageStats:
    """Compute the full metric set for a float RGB image."""
    h, w = img.shape[:2]
    small = _downscale_for_analysis(img)
    y = luma(small)

    brightness = float(np.median(y))
    contrast = float(np.std(y))
    p1, p99 = np.percentile(y, [1, 99])
    dynamic_range = float(p99 - p1)
    shadow_clip = float(np.mean(y < 0.02))
    highlight_clip = float(np.mean(y > 0.98))
    entropy = _entropy(y)

    noise_sigma = _estimate_noise(y)
    sharpness, blur_sigma = _estimate_sharpness(y, noise_sigma)

    lab = rgb_to_lab(small)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    saturation = float(np.clip(np.mean(chroma) / 128.0, 0, 1))
    colorfulness = _colorfulness(small)
    gains, cast = _white_balance_gains(small)

    stats = ImageStats(
        width=w,
        height=h,
        megapixels=round(w * h / 1e6, 3),
        brightness=brightness,
        contrast=contrast,
        dynamic_range=dynamic_range,
        shadow_clip=shadow_clip,
        highlight_clip=highlight_clip,
        entropy=entropy,
        sharpness=sharpness,
        blur_sigma=blur_sigma,
        noise_sigma=noise_sigma,
        saturation=saturation,
        colorfulness=colorfulness,
        wb_cast=cast,
        wb_gains=gains,
    )
    stats.flags = _flags(stats)
    return stats


# --------------------------------------------------------------------------
# individual metrics
# --------------------------------------------------------------------------

def _downscale_for_analysis(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= _ANALYSIS_LONG_EDGE:
        return img
    scale = _ANALYSIS_LONG_EDGE / long_edge
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _entropy(y: np.ndarray) -> float:
    hist = np.histogram(y, bins=256, range=(0.0, 1.0))[0].astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _estimate_noise(y: np.ndarray) -> float:
    """Immerkaer's fast noise estimator.

    Convolves with a kernel that annihilates locally-linear signal, so what is
    left is dominated by noise. Robust enough to drive an auto-denoise strength.
    """
    h, w = y.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(y, cv2.CV_32F, kernel)
    conv = conv[1:-1, 1:-1]
    # Median absolute deviation rather than mean: keeps strong edges from
    # masquerading as noise.
    mad = float(np.median(np.abs(conv)))
    sigma = mad / 0.6745 * np.sqrt(np.pi / 2) / 6.0
    return float(np.clip(sigma, 0.0, 0.5))


def _estimate_sharpness(y: np.ndarray, noise_sigma: float) -> tuple[float, float]:
    """Return (normalised sharpness 0..1, estimated blur sigma in px).

    Laplacian variance alone is famously scale- and content-dependent, so it is
    normalised by local signal energy and corrected for the noise floor -- a
    grainy photo should not read as sharp.
    """
    # Noise masquerades as detail, and on a genuinely noisy frame the noise
    # term alone can exceed the whole Laplacian energy -- subtracting it then
    # leaves zero and every measurement downstream collapses. So suppress the
    # noise first, at a radius scaled to how much of it there is, and measure
    # on that. The pre-blur is accounted for when solving for blur radius.
    pre = 0.0 if noise_sigma < 0.006 else float(min(1.2, noise_sigma * 40.0))
    yp = y if pre == 0.0 else cv2.GaussianBlur(y, (0, 0), pre)

    energy = _detail_energy(yp)
    signal = float(np.var(y)) + 1e-6

    # Saturating rather than clipped: a clipped metric pins at 1.0 for any
    # reasonably detailed image and then cannot tell sharp from sharper, which
    # makes it useless for driving auto-sharpening.
    t = float(np.sqrt(energy / signal))
    normalised = float(t / (t + 0.55))

    blur_sigma = _estimate_blur_sigma(yp, energy, pre)
    return normalised, blur_sigma


def _detail_energy(y: np.ndarray) -> float:
    """Laplacian energy with the noise contribution removed.

    A 3x3 Laplacian amplifies white noise variance by a factor of 20 (the sum
    of its squared taps), so the noise floor is directly computable from the
    measured sigma rather than guessed.
    """
    raw = float(np.var(cv2.Laplacian(y, cv2.CV_32F, ksize=3)))
    return max(raw - (_estimate_noise(y) ** 2) * 20.0, 1e-9)


def _estimate_blur_sigma(y: np.ndarray, energy: float, pre_blur: float,
                         probe: float = 1.0) -> float:
    """Estimate the Gaussian blur already present, in pixels.

    Re-blur by a known amount and see how much high-frequency energy that
    destroys. An already-soft image barely notices; a crisp one loses most of
    it. Modelling both as Gaussians, Laplacian energy falls as sigma^-4, so

        r = E(after) / E(before) = (s0^2 / (s0^2 + probe^2))^2

    which inverts to a closed form for s0. Note this measures *relative* decay,
    which is what makes it work at all -- comparing an image's absolute energy
    against blurred copies of itself is circular and always reports zero.

    Known limitation: on heavily noisy frames this under-reports, sometimes to
    zero, because residual noise props up the post-probe energy. That is the
    safe direction -- the pipeline only ever uses this number to decide how
    hard to deconvolve, and deconvolving a noisy image is the thing you least
    want to do by accident. Trust ``sharpness`` instead when noise is high.
    """
    if energy <= 1e-8:
        return 3.0
    e1 = _detail_energy(cv2.GaussianBlur(y, (0, 0), probe))

    r = float(np.clip(e1 / energy, 1e-6, 0.999))
    root = np.sqrt(r)
    measured = probe * np.sqrt(root / max(1.0 - root, 1e-6))

    # Remove the blur we added ourselves to suppress noise, then the ~0.5px a
    # perfectly crisp frame still shows from sensor anti-aliasing.
    s0 = np.sqrt(max(measured ** 2 - pre_blur ** 2, 0.0))
    return float(np.clip(s0 - 0.5, 0.0, 6.0))


def _colorfulness(img: np.ndarray) -> float:
    """Hasler & Susstrunk (2003), rescaled so ~1.0 is vividly colourful."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(np.var(rg) + np.var(yb))
    mean = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(np.clip((std + 0.3 * mean) * 3.0, 0.0, 1.0))


def _white_balance_gains(img: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Shades-of-grey illuminant estimate (Minkowski p=6).

    More robust than grey-world on images with a large saturated region (a red
    car, a blue sky) which is exactly where grey-world embarrasses itself.
    """
    p = 6.0
    flat = np.clip(img.reshape(-1, 3), 1e-6, 1.0)
    # Ignore clipped pixels: they carry no illuminant information.
    mask = np.all(flat < 0.98, axis=1)
    if mask.sum() > 64:
        flat = flat[mask]
    illum = np.power(np.mean(np.power(flat, p), axis=0), 1.0 / p)
    illum = illum / (illum.mean() + 1e-8)
    gains = 1.0 / np.clip(illum, 1e-3, None)
    # Normalise so the gains are luminance-neutral. Scaling to the max instead
    # (the obvious choice) only ever pulls channels down, which silently
    # darkens every image that gets white-balanced.
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    gains = gains / max(float(np.dot(gains, weights)), 1e-6)
    cast = float(np.clip(np.std(illum) * 3.0, 0.0, 1.0))
    return (float(gains[0]), float(gains[1]), float(gains[2])), cast


def _flags(s: ImageStats) -> list[str]:
    out: list[str] = []
    if s.brightness < 0.34:
        out.append("underexposed")
    if s.brightness > 0.68:
        out.append("overexposed")
    if s.contrast < 0.10:
        out.append("flat/low-contrast")
    if s.dynamic_range < 0.45:
        out.append("hazy")
    if s.shadow_clip > 0.10:
        out.append("crushed-shadows")
    if s.highlight_clip > 0.06:
        out.append("blown-highlights")
    if s.blur_sigma > 0.9:
        out.append("soft-focus")
    if s.noise_sigma > 0.012:
        out.append("noisy")
    if s.wb_cast > 0.12:
        out.append("colour-cast")
    if s.saturation < 0.08:
        out.append("desaturated")
    if s.megapixels < 0.35:
        out.append("low-resolution")
    return out
