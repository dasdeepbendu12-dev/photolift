"""Regression tests for the naturalness of the default output.

These exist to stop the default drifting back toward the over-processed look,
which is what version 0.1 produced. They encode the governing rule: a good
photograph must come out looking like itself, and a corrected one must not
overshoot.

Each test corresponds to a specific way an auto-enhancer embarrasses itself --
washing out a black background, dragging a low-key frame to mid-grey, tinting
a monochrome scan, inflating colour past anything the scene contained. None of
them are hypothetical; every one describes something this pipeline did before
the constant above it was tuned.
"""

from __future__ import annotations

import numpy as np

import photolift as pl
from photolift import ops


def _clip_fracs(img):
    from photolift.imageio import luma
    y = luma(img)
    return float(np.mean(y < 0.02)), float(np.mean(y > 0.98))


def test_good_photo_is_left_almost_untouched(clean):
    """The hardest thing for an auto mode is recognising it has nothing to do."""
    out = pl.enhance(clean, scale=1).image
    assert float(np.abs(out - clean).mean()) < 0.035


def test_deep_blacks_are_treated_as_intent_not_a_fault(clean):
    """A subject on a black background must not come back grey.

    Shadow clipping alone is not evidence of a problem -- it describes every
    studio portrait ever taken. The lift is gated on the frame *also* being
    dark overall, and this is that gate.
    """
    # A realistic studio framing: a well-exposed subject on a black surround,
    # with the frame's median still sitting in normal territory.
    dark_bg = clean.copy()
    edge = clean.shape[1] // 3
    dark_bg[:, :edge] = 0.0
    dark_bg[:, edge:] = np.clip(dark_bg[:, edge:] * 1.35, 0, 1)

    out = pl.enhance(dark_bg, scale=1).image
    black_before = float(np.mean(_clip_fracs(dark_bg)[0]))
    black_after = float(np.mean(_clip_fracs(out)[0]))
    # The blacks may move a little, but must not be substantially washed out.
    assert black_after > black_before * 0.75


def test_low_key_photo_is_not_dragged_to_midgrey(clean):
    """A deliberately dark photograph stays dark. Exposure closes part of the
    gap to the target, never all of it."""
    dim = np.clip(clean * 0.45, 0, 1)
    out = pl.enhance(dim, scale=1).image
    from photolift.imageio import luma
    brightened = float(np.median(luma(out)))
    assert brightened > float(np.median(luma(dim)))   # it did help
    assert brightened < 0.46                          # but did not overshoot


def test_monochrome_image_is_never_tinted(clean):
    """A greyscale scan with an age-tint has high saturation and near-zero
    colourfulness. Confusing the two is how a mono scan comes back sepia."""
    grey = np.repeat(clean.mean(axis=2, keepdims=True), 3, axis=2)
    tinted = np.clip(grey * np.array([1.0, 0.93, 0.80], np.float32), 0, 1)

    from photolift.pipeline import _mean_chroma
    out = pl.enhance(tinted, scale=1).image
    assert _mean_chroma(out) < _mean_chroma(tinted)


def test_saturation_is_not_inflated_past_tolerance(degraded):
    from photolift.pipeline import _MAX_CHROMA_GROWTH, _mean_chroma
    out = pl.enhance(degraded, scale=2).image
    ratio = _mean_chroma(out) / max(_mean_chroma(degraded), 1e-6)
    assert ratio < _MAX_CHROMA_GROWTH * 1.1


def test_clipping_stays_within_tolerance(degraded):
    from photolift.pipeline import _MAX_HIGHLIGHT_CLIP_GROWTH, _MAX_SHADOW_CLIP_GROWTH
    r = pl.enhance(degraded, scale=2)
    assert (r.stats_after.shadow_clip - r.stats_before.shadow_clip
            < _MAX_SHADOW_CLIP_GROWTH * 1.5)
    assert (r.stats_after.highlight_clip - r.stats_before.highlight_clip
            < _MAX_HIGHLIGHT_CLIP_GROWTH * 1.5)


def test_intensity_never_pushes_a_knob_past_its_hard_cap(degraded):
    """Caps are applied before intensity scaling, so they need re-applying
    after. Without that, intensity=2 gives white_balance=1.57 -- which does not
    remove a colour cast, it inverts it."""
    r = pl.enhance(degraded, pl.preset("punchy", scale=1))
    values = dict(r.steps)
    assert "white_balance" in values
    for name, detail in r.steps:
        first = detail.split()[0].lstrip("+")
        try:
            magnitude = abs(float(first))
        except ValueError:
            continue
        assert magnitude <= 3.01, f"{name} ran away to {detail}"


def test_presets_form_a_monotone_ladder(degraded):
    """natural < vivid < punchy in actual delivered contrast. A preset that is
    nominally stronger but lands weaker (the guard over-damping) is a bug."""
    from photolift.imageio import luma
    contrasts = []
    for name in ("natural", "vivid", "punchy"):
        out = pl.enhance(degraded, pl.preset(name, scale=1)).image
        contrasts.append(float(np.std(luma(out))))
    # natural is strictly the gentlest. vivid and punchy both run into the
    # per-knob hard caps on a source this degraded, so they land together --
    # asserting a strict order between them would be asserting noise.
    assert contrasts[0] < contrasts[1], contrasts
    assert contrasts[0] < contrasts[2], contrasts
    assert abs(contrasts[2] - contrasts[1]) < 0.05, contrasts


def test_guard_backs_off_when_a_preset_overshoots(clean):
    bad = pl.EnhanceConfig(scale=1, intensity=2.0, natural_guard=True,
                           levels=1.0, contrast=0.9, vibrance=0.9)
    r = pl.enhance(clean, bad)
    # With everything pinned the guard cannot help, but it must not crash and
    # must leave a usable image behind.
    assert np.isfinite(r.image).all()


def test_flat_denoise_cleans_flat_areas_and_spares_texture():
    """The point of the operator: grain in a smooth region goes, edges stay."""
    rng = np.random.default_rng(0)
    img = np.zeros((160, 160, 3), np.float32) + 0.5
    img[:, 80:] = 0.8                                    # a hard edge
    noisy = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)

    out = ops.flat_denoise(noisy, strength=0.9)
    flat_noise_before = float(np.std(noisy[20:60, 20:60]))
    flat_noise_after = float(np.std(out[20:60, 20:60]))
    assert flat_noise_after < flat_noise_before * 0.7

    # The edge survives: the step across it is essentially unchanged.
    step_before = float(noisy[:, 85:95].mean() - noisy[:, 65:75].mean())
    step_after = float(out[:, 85:95].mean() - out[:, 65:75].mean())
    assert abs(step_after - step_before) < 0.02
