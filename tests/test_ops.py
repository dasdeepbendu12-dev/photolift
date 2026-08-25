"""Tests for the individual operators.

Each operator is usable standalone, so each is tested standalone. Two
properties matter across all of them and are tested exhaustively rather than
per-operator: a true no-op at zero strength, and a valid in-range float32
result at full strength.
"""

from __future__ import annotations

import numpy as np
import pytest

import photolift as pl
from photolift import ops


@pytest.mark.parametrize("fn,kwargs", [
    (ops.white_balance, {"strength": 0.0}),
    (ops.auto_exposure, {"strength": 0.0}),
    (ops.shadows_highlights, {"shadows": 0.0, "highlights": 0.0}),
    (ops.local_contrast, {"clip_limit": 0.0}),
    (ops.levels, {"strength": 0.0}),
    (ops.dehaze, {"strength": 0.0}),
    (ops.denoise, {"strength": 0.0}),
    (ops.deconvolve, {"sigma": 0.0}),
    (ops.unsharp_mask, {"amount": 0.0}),
    (ops.clarity, {"amount": 0.0}),
    (ops.vibrance, {"amount": 0.0}),
    (ops.tone_curve, {"contrast": 0.0}),
    (ops.add_grain, {"amount": 0.0}),
])
def test_every_operator_is_a_noop_at_zero(clean, fn, kwargs):
    assert np.allclose(fn(clean, **kwargs), clean, atol=1e-6)


@pytest.mark.parametrize("fn,kwargs", [
    (ops.white_balance, {"strength": 1.0}),
    (ops.auto_exposure, {"strength": 1.0}),
    (ops.shadows_highlights, {"shadows": 0.5, "highlights": 0.3}),
    (ops.local_contrast, {"clip_limit": 2.0}),
    (ops.dehaze, {"strength": 0.6}),
    (ops.deconvolve, {"sigma": 1.2}),
    (ops.unsharp_mask, {"amount": 1.0}),
    (ops.vibrance, {"amount": 0.4}),
    (ops.add_grain, {"amount": 0.02}),
])
def test_operators_stay_in_range_and_keep_shape(degraded, fn, kwargs):
    out = fn(degraded, **kwargs)
    assert out.shape == degraded.shape
    assert out.dtype == np.float32
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6


def test_exposure_moves_toward_target(degraded):
    out = ops.auto_exposure(degraded, target=0.5, strength=1.0)
    before = abs(pl.analyze(degraded).brightness - 0.5)
    assert abs(pl.analyze(out).brightness - 0.5) < before


def test_white_balance_reduces_cast(degraded):
    out = ops.white_balance(degraded, strength=1.0)
    assert pl.analyze(out).wb_cast < pl.analyze(degraded).wb_cast


def test_denoise_lowers_measured_noise(degraded):
    out = ops.denoise(degraded, strength=0.8)
    assert pl.analyze(out).noise_sigma < pl.analyze(degraded).noise_sigma


def test_sharpen_raises_measured_sharpness(clean):
    import cv2
    soft = cv2.GaussianBlur(clean, (0, 0), 1.2)
    out = ops.unsharp_mask(soft, amount=1.0)
    assert pl.analyze(out).sharpness > pl.analyze(soft).sharpness


def test_deconvolve_recovers_detail_without_exploding(clean):
    import cv2
    blurred = cv2.GaussianBlur(clean, (0, 0), 1.2)
    out = ops.deconvolve(blurred, sigma=1.2, iterations=15)
    assert pl.analyze(out).sharpness > pl.analyze(blurred).sharpness
    # halo control must keep it inside the valid range
    assert out.max() <= 1.0 + 1e-6


def test_luminance_operators_do_not_shift_hue(degraded):
    """Sharpening and contrast work on luma only; mean hue must survive."""
    from photolift.imageio import rgb_to_lab
    before = rgb_to_lab(degraded)
    after = rgb_to_lab(ops.unsharp_mask(degraded, amount=1.0))
    hue_before = np.arctan2(before[..., 2].mean(), before[..., 1].mean())
    hue_after = np.arctan2(after[..., 2].mean(), after[..., 1].mean())
    assert abs(hue_before - hue_after) < 0.05
