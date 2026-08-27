"""Before/after visualisations -- the only honest way to judge an enhancement."""

from __future__ import annotations

import cv2
import numpy as np

from .imageio import to_float

__all__ = ["side_by_side", "split_view", "zoom_strip"]

_LABEL_H = 34


def side_by_side(before: np.ndarray, after: np.ndarray, gap: int = 12,
                 label: bool = True) -> np.ndarray:
    """Both frames at the same display height, original on the left."""
    a, b = to_float(before), to_float(after)
    h = max(a.shape[0], b.shape[0])
    a = _fit_height(a, h)
    b = _fit_height(b, h)

    pad = np.ones((h, gap, 3), np.float32)
    canvas = np.concatenate([a, pad, b], axis=1)
    if label:
        canvas = _stack_labels(canvas, [("BEFORE", 0), ("AFTER", a.shape[1] + gap)])
    return canvas


def split_view(before: np.ndarray, after: np.ndarray, position: float = 0.5,
               line: bool = True) -> np.ndarray:
    """One frame, wiped down the middle. Best for spotting tone shifts, since
    both halves sit in the same visual context."""
    a, b = to_float(before), to_float(after)
    if a.shape[:2] != b.shape[:2]:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    x = int(np.clip(position, 0.0, 1.0) * b.shape[1])
    out = b.copy()
    out[:, :x] = a[:, :x]
    if line and 0 < x < out.shape[1]:
        out[:, max(0, x - 1):x + 1] = 1.0
    return out


def zoom_strip(before: np.ndarray, after: np.ndarray, box: tuple[int, int, int, int]
               | None = None, zoom: int = 3) -> np.ndarray:
    """100%-ish crop of both frames. Sharpening and denoising claims live or die
    here; a downscaled full frame hides everything that matters."""
    a, b = to_float(before), to_float(after)
    scale_y = b.shape[0] / a.shape[0]
    scale_x = b.shape[1] / a.shape[1]

    if box is None:
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        side = max(32, min(a.shape[0], a.shape[1]) // 5)
        box = (cx - side // 2, cy - side // 2, side, side)
    x, y, w, h = box

    crop_a = a[y:y + h, x:x + w]
    crop_b = b[int(y * scale_y):int((y + h) * scale_y),
               int(x * scale_x):int((x + w) * scale_x)]
    size = (w * zoom, h * zoom)
    crop_a = cv2.resize(crop_a, size, interpolation=cv2.INTER_NEAREST)
    crop_b = cv2.resize(crop_b, size, interpolation=cv2.INTER_NEAREST)
    return side_by_side(crop_a, crop_b)


def _fit_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * h / img.shape[0]))
    interp = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp)


def _stack_labels(canvas: np.ndarray, labels: list[tuple[str, int]]) -> np.ndarray:
    """Render the caption bar and stack it above the canvas.

    The text is drawn on an 8-bit buffer and converted afterwards, rather than
    straight onto the float canvas everything else here uses. That is not a
    style choice: OpenCV 5 asserts ``img.depth() == CV_8U`` inside ``putText``,
    where 4.x quietly accepted a float image. Drawing on float therefore works
    on a pinned old install and fails on every machine that resolves a current
    OpenCV -- which is the worst possible failure mode, since it passes locally
    and breaks for everyone else.
    """
    bar = np.full((_LABEL_H, canvas.shape[1], 3), 20, np.uint8)      # ~0.08
    for text, x in labels:
        cv2.putText(bar, text, (x + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (242, 242, 242), 1, cv2.LINE_AA)                 # ~0.95
    return np.concatenate([bar.astype(np.float32) / 255.0, canvas], axis=0)
