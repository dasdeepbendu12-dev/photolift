"""The enhancement pipeline: order of operations, auto-strength, presets."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from dataclasses import replace as _replace
from typing import Callable

import numpy as np

from . import ops
from .analysis import ImageStats, analyze
from .imageio import load, save, to_float
from .upscale import upscale

__all__ = ["EnhanceConfig", "Result", "Enhancer", "enhance", "enhance_file",
           "enhance_batch", "PRESETS", "preset"]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class EnhanceConfig:
    """Every knob, in the order the pipeline applies them.

    With ``auto=True`` (the default) any value left at ``None`` is chosen from
    the measured image statistics. Set a value explicitly to override just that
    step and let the rest stay automatic -- that mix is the point of the design.
    """

    auto: bool = True

    # Global authority of the automatic corrections. 1.0 is tuned so that a
    # good photograph comes back looking like itself; raise it for a punchier,
    # more obviously "enhanced" result, lower it for a near-invisible touch-up.
    # Measured corrections (denoise, deconvolve) are not scaled by this --
    # noise is a defect, not a style.
    intensity: float = 1.0

    # Safety net: after the pipeline runs, compare the result against the input
    # and back off automatically if it crushed shadows, blew highlights, or
    # oversaturated. Costs one extra pass when it actually triggers.
    natural_guard: bool = True

    # resolution
    scale: float = 2.0
    max_megapixels: float | None = 80.0     # guard against accidental 400MP jobs
    backend: str = "auto"
    model_path: str | None = None
    tile: int | None = 512

    # colour / lighting
    white_balance: float | None = None      # 0..1 blend toward neutral
    exposure: float | None = None           # 0..1 strength toward target
    exposure_target: float = 0.48
    shadows: float | None = None            # 0..1 lift
    highlights: float | None = None         # 0..1 recovery
    dehaze: float | None = None             # 0..1

    # contrast
    levels: float | None = None             # 0..1 black/white point stretch
    local_contrast: float | None = None     # CLAHE clip limit, 0..4
    contrast: float | None = None           # S-curve, -1..1
    clarity: float | None = None            # midtone punch, 0..1

    # detail
    denoise: float | None = None            # 0..1
    flat_denoise: float | None = None       # 0..1, post-upscale cleanup
    deconvolve: float | None = None         # PSF sigma in px, 0 disables
    deconvolve_iterations: int = 12
    sharpen: float | None = None            # unsharp amount, 0..2
    sharpen_radius: float | None = None
    sharpen_threshold: float | None = None   # edge floor; auto-set from noise

    # finishing
    vibrance: float | None = None           # -1..1
    grain: float | None = None              # 0..0.05
    seed: int | None = None

    # output
    quality: int = 95
    bit_depth: int | None = None

    def merged(self, **overrides) -> EnhanceConfig:
        clean = {k: v for k, v in overrides.items() if v is not None}
        return _replace(self, **clean)


@dataclass
class Result:
    """What came out, and what was done to get there."""

    image: np.ndarray
    stats_before: ImageStats
    stats_after: ImageStats
    steps: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float = 0.0
    path: str | None = None

    def report(self) -> str:
        lines = ["before: " + self.stats_before.summary(),
                 "after:  " + self.stats_after.summary(),
                 f"steps ({self.elapsed:.2f}s):"]
        lines += [f"  - {name}: {detail}" for name, detail in self.steps] or ["  (none)"]
        return "\n".join(lines)

    def save(self, path: str, quality: int = 95,
             bit_depth: int | None = None) -> str:
        return save(self.image, path, quality=quality, bit_depth=bit_depth)


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    # The default, and the one tuned hardest. Fixes what is measurably wrong,
    # in proportion to how wrong it is, and stops. A good photo comes back
    # looking like itself.
    "natural": {},
    # Slightly more authority across the board, still measurement-driven. For
    # when "natural" is too polite for a genuinely flat source.
    "vivid": {"intensity": 1.6},
    # Punchy, deliberately processed. This is the look the pipeline used to
    # produce by default; kept as an explicit choice rather than a surprise.
    "punchy": {"intensity": 2.0, "natural_guard": False},
    # Skin first: light sharpening, gentle chroma denoise, no clarity crunch.
    "portrait": {"intensity": 0.8, "sharpen": 0.35, "sharpen_threshold": 0.02,
                 "clarity": 0.08, "vibrance": 0.08, "local_contrast": 0.8,
                 "grain": 0.004},
    # Detail and depth: more clarity, dehaze on, colour pushed.
    "landscape": {"intensity": 1.4, "clarity": 0.30, "dehaze": 0.35,
                  "vibrance": 0.20, "local_contrast": 1.4, "sharpen": 0.7},
    # Text and line art: legibility beats naturalness, so the guard is off and
    # clipping the page to white is the point.
    "document": {"denoise": 0.25, "levels": 1.0, "local_contrast": 2.5,
                 "sharpen": 1.1, "sharpen_threshold": 0.005, "vibrance": -0.2,
                 "grain": 0.0, "contrast": 0.25, "natural_guard": False},
    # High-ISO night frames: denoise hard, lift shadows, hold saturation back.
    "lowlight": {"denoise": 0.75, "shadows": 0.4, "highlights": 0.2,
                 "vibrance": 0.12, "sharpen": 0.4, "grain": 0.008,
                 "local_contrast": 1.0},
    # For genuinely bad source material. Will look processed. That is the deal,
    # so the guard is off -- it would only fight the intent.
    "aggressive": {"intensity": 2.0, "natural_guard": False,
                   "denoise": 0.7, "levels": 1.0, "local_contrast": 2.6,
                   "clarity": 0.6, "sharpen": 1.2, "vibrance": 0.35,
                   "dehaze": 0.6, "contrast": 0.3},
    # Change resolution only. Useful as a control when judging the rest.
    "resize_only": {"white_balance": 0.0, "exposure": 0.0, "shadows": 0.0,
                    "highlights": 0.0, "dehaze": 0.0, "levels": 0.0,
                    "local_contrast": 0.0, "contrast": 0.0, "clarity": 0.0,
                    "denoise": 0.0, "flat_denoise": 0.0, "deconvolve": 0.0,
                    "sharpen": 0.0, "vibrance": 0.0, "grain": 0.0},
}


def preset(name: str, **overrides) -> EnhanceConfig:
    """Build a config from a named preset, with optional per-call overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    return EnhanceConfig(**{**PRESETS[name], **overrides})


# --------------------------------------------------------------------------
# auto-strength
# --------------------------------------------------------------------------

def _sharpen_amount(s: ImageStats, scale: float) -> float:
    """Output sharpening strength: inverse to measured sharpness, with a floor
    (upscaling softens by construction, so some is always warranted).

    The pivot sits at 0.75 rather than 1.0 because the sharpness metric
    saturates well below 1.0 on real images; pivoting at 1.0 would declare
    every photo sharp enough and the auto-sharpener would never fire.
    """
    amount = float(np.clip((0.75 - s.sharpness) * 1.6, 0.2, 1.0))
    if scale > 1.0:
        amount *= 1.0 + min(scale - 1.0, 3.0) * 0.18
    # Never sharpen hard into noise -- that just makes the grain crisper.
    amount *= float(np.clip(1.0 - s.noise_sigma * 18.0, 0.4, 1.0))
    return float(np.clip(amount, 0.0, 1.4))


def _autofill(cfg: EnhanceConfig, s: ImageStats) -> EnhanceConfig:
    """Turn measurements into strengths.

    The governing principle is that **a photograph that is already good must
    come out almost unchanged**. Every correction's authority is proportional
    to how far the measurement actually sits outside the normal range, with a
    generous deadband before it engages at all, and a cap well short of a full
    correction. A partial fix reads as a better photograph; a complete fix
    reads as a processed one.

    Concretely, that means trusting the photographer by default:

      * a low-key photo is allowed to be dark -- ``exposure`` moves it part of
        the way toward the target, never all the way;
      * deep blacks are usually a choice, not a fault, so ``shadows`` only
        lifts when the frame is *also* dark overall;
      * a warm photo is allowed to stay warm -- ``white_balance`` caps at a
        partial neutralisation;
      * a monochrome image stays monochrome.

    Every constant here is scaled by ``cfg.intensity``, so the punchier look is
    one argument away without any of these decisions being relitigated.
    """
    if not cfg.auto:
        return cfg
    d: dict = {}
    k = float(np.clip(cfg.intensity, 0.0, 2.0))

    def put(key: str, value, deadband: float = 0.02, scale: bool = True,
            hard_max: float = 1.0) -> None:
        """Set ``key`` unless the user pinned it.

        Values below ``deadband`` snap to zero: a 0.01-strength correction is
        invisible and only costs a pipeline stage, and a step list full of
        no-ops makes the report useless for understanding what happened. The
        deadbands are the main thing keeping a good photo unmolested, so they
        are set from what a *normal* image measures, not from what is visible.

        ``hard_max`` is re-applied *after* the intensity scaling, and that
        order is not incidental. The per-knob caps computed above are clipped
        before scaling, so without a second clip an ``intensity`` above 1.0
        multiplies straight past them -- which is how you get a white balance
        strength of 1.57, meaning the cast is not merely removed but inverted.
        """
        if getattr(cfg, key) is None:
            if isinstance(value, float):
                if scale:
                    value *= k
                value = float(np.clip(value, -hard_max, hard_max))
                if abs(value) < deadband:
                    value = 0.0
            d[key] = value

    # How much genuine colour *variety* the image has, 0..1. Note this is
    # colourfulness (the spread of hues present), not saturation (how strong
    # they are): a greyscale scan with a yellow age-tint has high saturation
    # and near-zero colourfulness, and telling those apart is exactly what is
    # needed here.
    #
    # It gates the two colour decisions in opposite directions, and both
    # follow from one observation -- a cast on an image with no colour variety
    # cannot be an artistic choice, because there are no colours to have
    # chosen. So such an image gets its cast removed *completely* rather than
    # partially, and gets no saturation push at all.
    # The offset matters: a noisy monochrome scan measures ~0.18 colourfulness
    # purely from chroma noise, so a plain ratio would score it half-colourful
    # and happily saturate the noise. Subtracting the floor first puts it near
    # zero where it belongs, while any real colour photo saturates the gate.
    colour_variety = float(np.clip((s.colorfulness - 0.12) / 0.28, 0.0, 1.0))
    monochrome = s.colorfulness < 0.02

    # The caps below are *severity-proportional*, and that is the whole trick.
    # A flat cap tuned to protect good photographs also refuses to rescue bad
    # ones; a flat cap tuned to rescue bad ones mangles good ones. So the
    # ceiling rises only once the measurement is far enough outside normal that
    # the defect cannot plausibly be an artistic choice.

    # --- colour cast. A mild cast is golden hour, tungsten, or a deliberate
    #     grade, so only the edge comes off. A severe one is a broken white
    #     balance, and nobody chose it.
    wb_cap = float(np.clip(0.55 + (s.wb_cast - 0.30) * 1.2, 0.55, 0.95))
    wb_cap = 0.98 + (wb_cap - 0.98) * colour_variety   # -> 0.98 for a mono scan
    wb_base = float(np.clip((s.wb_cast - 0.10) * 2.2, 0.0, 1.0))
    # Multiplied, not offset: an image with no cast still gets no correction,
    # however little colour variety it has.
    put("white_balance",
        0.0 if monochrome else float(np.clip(wb_base * (1.0 + 2.0 *
                                                        (1.0 - colour_variety)),
                                             0.0, wb_cap)),
        deadband=0.06, hard_max=1.0)

    # --- exposure. Wide deadband; a moody frame stays moody. But a frame three
    #     stops down is not moody, it is underexposed.
    off = abs(s.brightness - cfg.exposure_target)
    exp_cap = float(np.clip(0.55 + (off - 0.20) * 1.5, 0.55, 0.85))
    put("exposure", float(np.clip((off - 0.09) * 2.2, 0.0, exp_cap)), deadband=0.05,
        hard_max=0.95)

    # --- shadows. Crushed blacks are usually intent (a dark background, a
    #     silhouette), so require corroborating evidence that the *frame* is
    #     dark before lifting. Without this gate a studio portrait on black
    #     comes back grey and flat.
    dark_frame = float(np.clip((0.42 - s.brightness) * 3.0, 0.0, 1.0))
    put("shadows",
        float(np.clip(max(0.0, s.shadow_clip - 0.05) * 1.6, 0.0, 0.30)) * dark_frame,
        deadband=0.04, hard_max=0.60)
    # Blown highlights are almost never intentional, so this one gets to act
    # on its own evidence -- but recovery beyond a point invents grey mush.
    put("highlights", float(np.clip(max(0.0, s.highlight_clip - 0.01) * 2.0,
                                    0.0, 0.35)), deadband=0.04, hard_max=0.55)

    # --- haze: compressed dynamic range *and* low entropy together. Either one
    #     alone describes plenty of perfectly good photographs (a foggy morning,
    #     a minimalist composition), so require both.
    hazy = (max(0.0, 0.42 - s.dynamic_range) * 1.3
            * max(0.0, min(1.0, (6.6 - s.entropy) * 0.5)))
    put("dehaze", float(np.clip(hazy, 0.0, 0.40)), deadband=0.08, hard_max=0.70)

    # --- contrast family. The deadband on levels matters: an image that
    #     already spans most of the range must not be "stretched" at all, or
    #     the black point walks and midtones darken for no benefit.
    levels_cap = float(np.clip(0.70 + (0.40 - s.dynamic_range) * 1.2, 0.70, 0.95))
    put("levels", float(np.clip((0.75 - s.dynamic_range) * 2.0, 0.0, levels_cap)),
        deadband=0.10, hard_max=1.0)

    # Local contrast and clarity are the two knobs that produce the "HDR phone
    # photo" look, so they are the most restrained. They are also noise
    # amplifiers -- they multiply whatever grain survived denoising -- so they
    # are damped again in proportion to how much there was.
    noise_damp = float(np.clip(1.0 - s.noise_sigma * 25.0, 0.35, 1.0))
    put("local_contrast",
        float(np.clip((0.13 - s.contrast) * 10.0, 0.0, 1.6)) * noise_damp,
        deadband=0.15, hard_max=3.0)
    put("contrast", float(np.clip((0.11 - s.contrast) * 1.6, 0.0, 0.22)),
        deadband=0.04, hard_max=0.45)
    put("clarity", float(np.clip((0.12 - s.contrast) * 1.5, 0.0, 0.28)) * noise_damp,
        deadband=0.05, hard_max=0.60)

    # --- denoise scaled to the measured sigma, and pulled back when the image
    #     is already soft (denoising a blurry frame just erases what is left).
    noise_strength = float(np.clip((s.noise_sigma - 0.004) * 55.0, 0.0, 0.85))
    if s.blur_sigma > 1.5:
        noise_strength *= 0.6
    # Denoise and deconvolve are not style choices -- they are corrections
    # against a measured physical defect -- so intensity does not scale them.
    put("denoise", noise_strength, scale=False)

    # Flat-area cleanup runs after the contrast stages have amplified the
    # residual grain, so it is driven by the *input* noise but sized for what
    # those stages will do to it.
    put("flat_denoise", float(np.clip((s.noise_sigma - 0.005) * 45.0, 0.0, 0.8)),
        scale=False, deadband=0.05, hard_max=0.9)

    # --- deconvolution: only when blur is real and noise is low enough that
    #     inverting the blur will not amplify grain into mush.
    if s.blur_sigma > 0.55 and s.noise_sigma < 0.03:
        put("deconvolve", float(np.clip(s.blur_sigma * 0.75, 0.5, 2.0)), scale=False)
    else:
        put("deconvolve", 0.0, scale=False)

    put("sharpen", float(np.clip(_sharpen_amount(s, cfg.scale) * k, 0.0, 1.6)),
        scale=False, hard_max=1.6)
    put("sharpen_radius",
        float(np.clip(0.8 + (cfg.scale - 1.0) * 0.35, 0.6, 2.2)), scale=False)
    # Raise the edge floor on a noisy source so sharpening skips flat areas
    # entirely. Without this the final pass puts back the grain that denoising
    # just spent real time removing.
    put("sharpen_threshold",
        float(np.clip(0.008 + s.noise_sigma * 1.2, 0.008, 0.045)), scale=False)

    # --- colour finishing. Expressed as a ratio toward a normal saturation
    #     rather than a difference, because "half as colourful as it should be"
    #     needs a much bigger push at low saturation than the same absolute gap
    #     needs at high saturation.
    #
    #     The target is deliberately a *normal* palette, not a vivid one. With
    #     no reference to compare against there is no way to know whether a
    #     given photo was richly coloured or muted to begin with, so aiming at
    #     ordinary and stopping is the only defensible choice. Images that
    #     really were vivid come back slightly under -- correctable with
    #     ``vibrance=`` or ``intensity=``, which is the right way round.
    sat_target = 0.185
    sat_gap = sat_target / max(s.saturation, 0.02) - 1.0
    vib_cap = float(np.clip(0.18 + (0.13 - s.saturation) * 3.0, 0.18, 0.50))
    put("vibrance",
        0.0 if monochrome else float(np.clip(sat_gap * 0.55, 0.0, vib_cap))
        * colour_variety,
        deadband=0.04, hard_max=0.55)

    # --- a whisper of grain only if we denoised or upscaled hard enough to
    #     have created a plastic surface.
    heavy = d.get("denoise", cfg.denoise or 0.0) > 0.4 or cfg.scale >= 3.0
    put("grain", 0.006 if heavy else 0.0, deadband=0.0, scale=False)

    return _replace(cfg, **d)


# --------------------------------------------------------------------------
# naturalness guard
# --------------------------------------------------------------------------

# Tolerances, in the units of the measurements themselves. Each allows a real
# improvement through and catches the corresponding way of ruining a photo.
_MAX_SHADOW_CLIP_GROWTH = 0.030    # fraction of pixels newly crushed to black
_MAX_HIGHLIGHT_CLIP_GROWTH = 0.015  # ...newly blown to white
_MAX_CHROMA_GROWTH = 1.35          # ratio of mean LAB chroma


def _overprocessing_excess(before: ImageStats, after: ImageStats,
                           before_img: np.ndarray, after_img: np.ndarray) -> float:
    """How far past 'improved' the result went, as a scalar in 0..1+.

    Zero means every measurement stayed inside tolerance. The three failures it
    catches are the three that cannot be undone by the viewer: detail destroyed
    at the black end, detail destroyed at the white end, and colour inflated
    past anything the scene contained.

    Note the asymmetry with :func:`_autofill` -- that guesses strengths from
    the input, this checks the actual output. Predicting clipping from input
    statistics alone is not reliable when six stages compose, so the honest
    thing is to measure the result and correct if it went too far.
    """
    shadow = max(0.0, (after.shadow_clip - before.shadow_clip)
                 - _MAX_SHADOW_CLIP_GROWTH) / _MAX_SHADOW_CLIP_GROWTH
    highlight = max(0.0, (after.highlight_clip - before.highlight_clip)
                    - _MAX_HIGHLIGHT_CLIP_GROWTH) / _MAX_HIGHLIGHT_CLIP_GROWTH

    c_before = _mean_chroma(before_img)
    c_after = _mean_chroma(after_img)
    ratio = c_after / max(c_before, 1e-6)
    chroma = max(0.0, ratio - _MAX_CHROMA_GROWTH) / _MAX_CHROMA_GROWTH

    return float(min(max(shadow, highlight, chroma), 4.0))


def _mean_chroma(img: np.ndarray) -> float:
    from .imageio import rgb_to_lab
    lab = rgb_to_lab(img)
    return float(np.mean(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)))


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

class Enhancer:
    """Reusable enhancer. Build once with a config, run over many images."""

    def __init__(self, config: EnhanceConfig | None = None,
                 progress: Callable[[str], None] | None = None) -> None:
        self.config = config or EnhanceConfig()
        self.progress = progress

    # -- public ------------------------------------------------------------

    def enhance(self, image: np.ndarray | str | os.PathLike, **overrides) -> Result:
        cfg = self.config.merged(**overrides)
        src_path = None
        if isinstance(image, (str, os.PathLike)):
            src_path = os.fspath(image)
            img = load(src_path)
        else:
            img = to_float(np.asarray(image))
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        start = time.perf_counter()
        before = analyze(img)
        # Remember what the caller pinned, before auto-fill obliterates the
        # distinction between "unset" and "computed".
        pinned = {f.name for f in fields(cfg) if getattr(cfg, f.name) is not None}
        base = self._clamp_scale(cfg, img)

        steps: list[tuple[str, str]] = []
        out = self._run(img, _autofill(base, before), steps, pinned)
        after = analyze(out)

        # --- naturalness guard -------------------------------------------
        # Auto-fill guesses strengths from the input; six stages then compose
        # in ways no per-stage guess fully predicts. So measure the result, and
        # if it over-processed, redo it with less authority. Runs at most twice.
        if base.auto and base.natural_guard:
            excess = _overprocessing_excess(before, after, img, out)
            if excess > 0.0:
                # Gentle, and floored well above zero. An aggressive damping
                # curve overshoots: it can land a high-intensity preset *below*
                # where plain "natural" would have put it, which is a worse
                # result than the one it was correcting.
                damped = float(np.clip(1.0 / (1.0 + 0.45 * excess), 0.55, 0.90))
                self._say(f"naturalness guard: excess {excess:.2f}, "
                          f"retrying at intensity x{damped:.2f}")
                retry_cfg = _autofill(
                    _replace(base, intensity=base.intensity * damped), before)
                retry_steps: list[tuple[str, str]] = []
                retry_out = self._run(img, retry_cfg, retry_steps, pinned)
                retry_after = analyze(retry_out)
                # Only keep the retry if it actually improved matters; a guard
                # that can make things worse is worse than no guard.
                if _overprocessing_excess(before, retry_after, img,
                                          retry_out) < excess:
                    out, after, steps = retry_out, retry_after, retry_steps
                    steps.append(("naturalness_guard",
                                  f"intensity x{damped:.2f} (excess {excess:.2f})"))

        return Result(
            image=out,
            stats_before=before,
            stats_after=after,
            steps=steps,
            elapsed=time.perf_counter() - start,
            path=src_path,
        )

    __call__ = enhance

    # -- internals ---------------------------------------------------------

    def _clamp_scale(self, cfg: EnhanceConfig, img: np.ndarray) -> EnhanceConfig:
        if not cfg.max_megapixels:
            return cfg
        h, w = img.shape[:2]
        out_mp = (h * cfg.scale) * (w * cfg.scale) / 1e6
        if out_mp <= cfg.max_megapixels:
            return cfg
        allowed = float(np.sqrt(cfg.max_megapixels * 1e6 / (h * w)))
        self._say(f"scale clamped {cfg.scale:g} -> {allowed:.2f} "
                  f"(max_megapixels={cfg.max_megapixels:g})")
        return _replace(cfg, scale=max(1.0, allowed))

    def _run(self, img: np.ndarray, cfg: EnhanceConfig,
             steps: list[tuple[str, str]], pinned: set[str]) -> np.ndarray:
        """Order is the whole argument.

        1. Neutralise colour and set exposure first, so every later step works
           on correctly-lit pixels.
        2. Denoise immediately after, and *before* anything that stretches
           contrast. Dehaze divides by a small transmission term and levels
           multiplies the shadows; run either on a noisy frame and you get
           blotchy chroma mottling in the darks that no later step can remove.
        3. Then the contrast work: shadows/highlights, dehaze, levels.
        4. Deconvolve at native resolution, where the blur model is valid.
        5. Upscale.
        6. Local contrast and clarity after upscaling, so their radii land on
           the final pixel grid rather than being magnified.
        7. Sharpen last, at output resolution -- output sharpening is the final
           step in every serious imaging chain for a reason.
        """
        def step(name: str, detail: str, fn) -> None:
            nonlocal img
            self._say(f"{name}: {detail}")
            img = fn(img)
            steps.append((name, detail))

        if cfg.white_balance:
            step("white_balance", f"{cfg.white_balance:.2f}",
                 lambda i: ops.white_balance(i, cfg.white_balance))

        if cfg.exposure:
            step("exposure", f"{cfg.exposure:.2f} -> target {cfg.exposure_target:.2f}",
                 lambda i: ops.auto_exposure(i, cfg.exposure_target, cfg.exposure))

        if cfg.denoise:
            step("denoise", f"{cfg.denoise:.2f}", lambda i: ops.denoise(i, cfg.denoise))

        if cfg.shadows or cfg.highlights:
            step("shadows_highlights",
                 f"shadows {cfg.shadows or 0:.2f} / highlights {cfg.highlights or 0:.2f}",
                 lambda i: ops.shadows_highlights(i, cfg.shadows or 0.0,
                                                  cfg.highlights or 0.0))

        if cfg.dehaze:
            step("dehaze", f"{cfg.dehaze:.2f}", lambda i: ops.dehaze(i, cfg.dehaze))

        if cfg.levels:
            step("levels", f"{cfg.levels:.2f}",
                 lambda i: ops.levels(i, strength=cfg.levels))

        if cfg.deconvolve:
            step("deconvolve",
                 f"sigma {cfg.deconvolve:.2f}, {cfg.deconvolve_iterations} iters",
                 lambda i: ops.deconvolve(i, cfg.deconvolve,
                                          cfg.deconvolve_iterations))

        if cfg.scale and abs(cfg.scale - 1.0) > 1e-6:
            step("upscale", f"x{cfg.scale:g} ({cfg.backend})",
                 lambda i: upscale(i, cfg.scale, cfg.backend, cfg.model_path,
                                   cfg.tile))

        if cfg.local_contrast:
            step("local_contrast", f"clip {cfg.local_contrast:.2f}",
                 lambda i: ops.local_contrast(i, cfg.local_contrast))

        if cfg.contrast:
            step("contrast", f"{cfg.contrast:+.2f}",
                 lambda i: ops.tone_curve(i, cfg.contrast))

        if cfg.clarity:
            step("clarity", f"{cfg.clarity:.2f}", lambda i: ops.clarity(i, cfg.clarity))

        if cfg.vibrance:
            step("vibrance", f"{cfg.vibrance:+.2f}",
                 lambda i: ops.vibrance(i, cfg.vibrance))

        if cfg.flat_denoise:
            step("flat_denoise", f"{cfg.flat_denoise:.2f}",
                 lambda i: ops.flat_denoise(i, cfg.flat_denoise))

        # Sharpening is a decision about the *output* pixel grid, so re-measure
        # here rather than trusting a number computed before upscaling and
        # local contrast changed the image underneath it.
        if cfg.auto and "sharpen" not in pinned:
            cfg = _replace(cfg, sharpen=_sharpen_amount(analyze(img), 1.0))

        if cfg.sharpen:
            step("sharpen",
                 f"amount {cfg.sharpen:.2f}, radius {cfg.sharpen_radius or 1.0:.2f}",
                 lambda i: ops.unsharp_mask(i, cfg.sharpen,
                                            cfg.sharpen_radius or 1.0,
                                            cfg.sharpen_threshold or 0.012))

        if cfg.grain:
            step("grain", f"{cfg.grain:.3f}",
                 lambda i: ops.add_grain(i, cfg.grain, seed=cfg.seed))

        return np.clip(img, 0.0, 1.0)

    def _say(self, message: str) -> None:
        if self.progress:
            self.progress(message)


# --------------------------------------------------------------------------
# functional API
# --------------------------------------------------------------------------

def enhance(image, config: EnhanceConfig | None = None, **overrides) -> Result:
    """One-shot enhance. ``image`` may be a path or an array."""
    return Enhancer(config).enhance(image, **overrides)


def enhance_file(src: str, dst: str, config: EnhanceConfig | None = None,
                 **overrides) -> Result:
    """Enhance ``src`` and write the result to ``dst``."""
    cfg = (config or EnhanceConfig()).merged(**overrides)
    result = Enhancer(cfg).enhance(src)
    result.save(dst, quality=cfg.quality, bit_depth=cfg.bit_depth)
    result.path = dst
    return result


def enhance_batch(
    sources: Iterable[str],
    out_dir: str,
    config: EnhanceConfig | None = None,
    suffix: str = "_enhanced",
    ext: str | None = None,
    on_result: Callable[[str, Result | Exception], None] | None = None,
    **overrides,
) -> list[Result]:
    """Process many files. One failure does not abort the run.

    ``on_result`` is called with ``(source, Result)`` on success and
    ``(source, Exception)`` on failure, so a caller can report progress and
    collect errors without the run stopping on a single corrupt file.
    """
    cfg = (config or EnhanceConfig()).merged(**overrides)
    enhancer = Enhancer(cfg)
    os.makedirs(out_dir, exist_ok=True)
    results: list[Result] = []

    for src in sources:
        stem, src_ext = os.path.splitext(os.path.basename(src))
        dst = os.path.join(out_dir, f"{stem}{suffix}{ext or src_ext}")
        try:
            result = enhancer.enhance(src)
            result.save(dst, quality=cfg.quality, bit_depth=cfg.bit_depth)
            result.path = dst
            results.append(result)
            if on_result:
                on_result(src, result)
        except Exception as exc:  # keep going: a batch of 500 should not die on 1
            if on_result:
                on_result(src, exc)
    return results
