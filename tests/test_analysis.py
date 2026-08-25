"""Tests for the measurement layer.

Every strength the pipeline chooses comes from these numbers, so a metric that
lies produces a plausible-looking pipeline that is wrong everywhere. The tests
here pin the properties the auto mode actually relies on: monotonicity in the
defect being measured, and immunity to the defects it is *not* measuring.
"""

from __future__ import annotations

import numpy as np
import pytest

import photolift as pl


def test_analysis_detects_the_defects_we_injected(clean, degraded):
    good = pl.analyze(clean)
    bad = pl.analyze(degraded)

    assert bad.brightness < good.brightness          # we underexposed it
    assert bad.contrast < good.contrast              # and flattened it
    assert bad.noise_sigma > good.noise_sigma        # and added noise
    assert bad.sharpness < good.sharpness            # and blurred it
    assert bad.wb_cast > good.wb_cast                # and warmed it

    assert "underexposed" in bad.flags
    assert "flat/low-contrast" in bad.flags
    assert "noisy" in bad.flags


@pytest.mark.parametrize("true_sigma", [1.0, 1.5, 2.0])
def test_blur_estimate_tracks_real_blur(clean, true_sigma):
    """On a clean frame the blur estimate must be monotone in actual blur --
    that is what makes it safe to hand to the deconvolver."""
    import cv2
    est = pl.analyze(cv2.GaussianBlur(clean, (0, 0), true_sigma)).blur_sigma
    baseline = pl.analyze(clean).blur_sigma
    assert est > baseline
    assert est < true_sigma + 1.0      # never wildly over -- over-estimating rings


def test_sharpness_is_not_fooled_by_noise(clean):
    """Noise is high-frequency energy but it is not detail. A metric that
    cannot tell them apart would auto-sharpen every grainy photo into mush."""
    rng = np.random.default_rng(0)
    noisy = np.clip(clean + rng.normal(0, 0.03, clean.shape).astype(np.float32), 0, 1)
    assert pl.analyze(noisy).sharpness < pl.analyze(clean).sharpness


def test_analysis_is_resolution_stable(clean):
    import cv2
    half = cv2.resize(clean, (128, 128), interpolation=cv2.INTER_AREA)
    a, b = pl.analyze(clean), pl.analyze(half)
    assert abs(a.brightness - b.brightness) < 0.05
    assert abs(a.contrast - b.contrast) < 0.05


def test_stats_roundtrip_to_dict(clean):
    d = pl.analyze(clean).as_dict()
    assert d["width"] == 256 and "sharpness" in d
