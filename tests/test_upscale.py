"""Tests for the resolution backends.

The interesting test here is the one that states what back-projection actually
promises -- consistency with the input under downsampling -- rather than the
one people reach for first (RMSE against ground truth, which rewards
blurriness and would score plain Lanczos higher).
"""

from __future__ import annotations

import numpy as np
import pytest

import photolift as pl


@pytest.mark.parametrize("backend", ["cubic", "lanczos", "iterative"])
def test_upscale_hits_exact_target_size(degraded, backend):
    out = pl.upscale(degraded, 2.0, backend=backend)
    assert out.shape[:2] == (degraded.shape[0] * 2, degraded.shape[1] * 2)


def test_fractional_scale(degraded):
    out = pl.upscale(degraded, 1.5, backend="lanczos")
    assert out.shape[0] == round(degraded.shape[0] * 1.5)


def test_scale_one_is_identity(degraded):
    assert pl.upscale(degraded, 1.0) is degraded


def test_backprojection_is_more_self_consistent_than_interpolation(clean):
    """The specific promise of back-projection, stated precisely.

    It does not promise lower RMSE against ground truth -- RMSE rewards
    blurriness, so plain Lanczos often wins that and it tells you nothing about
    perceived quality. What it promises is *consistency*: downsample the result
    and you should get the input back. That is testable, so test that.
    """
    import cv2
    small = cv2.resize(clean, (128, 128), interpolation=cv2.INTER_AREA)
    lanczos = pl.upscale(small, 2.0, backend="lanczos")
    ibp = pl.upscale(small, 2.0, backend="iterative")

    from photolift.imageio import srgb_to_linear
    lr = srgb_to_linear(small)

    def reprojection_error(up):
        # Measured in linear light, because that is the domain the resampling
        # model lives in; comparing gamma-encoded values here would be scoring
        # the estimator against a model it never claimed to fit.
        down = cv2.resize(srgb_to_linear(up), (128, 128), interpolation=cv2.INTER_AREA)
        return float(np.sqrt(np.mean((down - lr) ** 2)))

    assert reprojection_error(ibp) < reprojection_error(lanczos)
    # ...and it should also come out visibly crisper, not just consistent.
    assert pl.analyze(ibp).sharpness > pl.analyze(lanczos).sharpness


def test_dnn_backend_reports_missing_model_clearly(degraded):
    with pytest.raises(RuntimeError, match="model"):
        pl.upscale(degraded, 2.0, backend="dnn", model_path=None)


def test_unknown_backend_rejected(degraded):
    with pytest.raises(ValueError):
        pl.upscale(degraded, 2.0, backend="magic")


def test_available_backends_includes_classical():
    avail = pl.available_backends()
    assert avail["lanczos"] and avail["iterative"]
