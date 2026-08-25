"""Synthetic test images with known, controllable defects."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def reference_scene(size: int = 256) -> np.ndarray:
    """A deterministic scene with edges, gradients, texture and colour.

    Enough structure that sharpness, contrast and colour metrics all have
    something to bite on -- a flat gradient would make every test pass.
    """
    rng = np.random.default_rng(7)
    h = w = size
    img = np.zeros((h, w, 3), np.float32)

    # tonal gradient background
    grad = np.linspace(0.15, 0.85, w, dtype=np.float32)
    img[:] = grad[None, :, None]

    # hard-edged colour blocks (edges for sharpness metrics)
    img[h // 8: h // 3, w // 8: w // 2] = (0.85, 0.20, 0.18)
    img[h // 2: 5 * h // 6, w // 6: w // 2] = (0.15, 0.45, 0.80)
    cv2.circle(img, (3 * w // 4, h // 3), size // 8, (0.20, 0.70, 0.30), -1)

    # fine texture (survives or dies under denoise -- that is the point)
    texture = rng.random((h, w)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 0.7)
    ys, xs = slice(2 * h // 3, None), slice(2 * w // 3, None)
    img[ys, xs] = np.clip(
        img[ys, xs] + (texture[ys, xs, None] - 0.5) * 0.5, 0, 1,
    )

    # thin lines (high frequency)
    for x in range(w // 2, w, 6):
        cv2.line(img, (x, 3 * h // 4), (x, h - 1), (0.95, 0.95, 0.95), 1)

    return np.clip(img, 0, 1)


def degrade(img: np.ndarray, blur: float = 1.6, noise: float = 0.02,
            downscale: int = 2, exposure: float = 0.45,
            cast: tuple[float, float, float] = (1.0, 0.92, 0.78),
            flatten: float = 0.55, seed: int = 3) -> np.ndarray:
    """Apply a realistic degradation chain: low light, warm cast, low contrast,
    blur, sensor noise, then downsampling. Roughly what a bad phone photo is."""
    rng = np.random.default_rng(seed)
    out = img * exposure                                   # underexpose
    out = 0.5 + (out - 0.5) * flatten                      # crush contrast
    out = out * np.asarray(cast, np.float32)               # colour cast
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)          # soft focus
    if downscale > 1:
        h, w = out.shape[:2]
        out = cv2.resize(out, (w // downscale, h // downscale),
                         interpolation=cv2.INTER_AREA)     # lose resolution
    if noise > 0:
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


@pytest.fixture
def clean() -> np.ndarray:
    return reference_scene()


@pytest.fixture
def degraded(clean: np.ndarray) -> np.ndarray:
    return degrade(clean)
