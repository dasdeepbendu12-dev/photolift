"""Naturalness benchmark: how *processed* does the output look?

This is the harness the v0.2 tuning was done against. It is not a test -- it
reports rather than asserts, it is slow, and it needs ``scikit-image`` for its
sample photographs -- so it lives here rather than in ``tests/``. The
regression tests in ``tests/test_naturalness.py`` lock in the thresholds this
benchmark was used to find.

    pip install -e ".[dev]"
    python benchmarks/eval_natural.py baseline
    # ...change something...
    python benchmarks/eval_natural.py candidate

The central design decision is what to score against. Every metric here
compares the result to the **original** photograph, not to the degraded input.
Scoring against the input conflates two opposite things -- legitimately
restoring contrast the degradation removed, and inflating contrast past
anything that was ever there. Only the second one looks unnatural, and only a
ground-truth comparison can tell them apart. So the benchmark degrades a known
original by a known amount and asks how close the pipeline gets back.

Every ratio below has an ideal value of 1.0 and every difference an ideal of
0.0. The most diagnostic rows are the ``already_ok`` ones, where the input *is*
the original: any deviation there is pure over-processing, with nothing to fix.

Metrics
-------
shadow_clip_vs_orig     detail destroyed at the black end
highlight_clip_vs_orig  detail destroyed at the white end
sat_delta_vs_orig       colour inflated (or drained) vs the original
halo                    overshoot energy along strong edges
local_std_vs_orig       micro-contrast inflation -- the "crunchy" look
skin_shift              hue movement in skin-tone pixels
rmse_vs_orig            overall fidelity, as a sanity anchor
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

import photolift as pl
from photolift.imageio import luma, rgb_to_lab

# The degradation model lives with the tests, since both need the same one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import degrade  # noqa: E402

try:
    from skimage import data
except ImportError:  # pragma: no cover
    raise SystemExit(
        "benchmarks need sample photographs: pip install -e '.[dev]'"
    ) from None


SCENES = {
    "astronaut": data.astronaut,   # portrait, skin, saturated orange suit
    "coffee": data.coffee,         # product, warm wood, specular highlights
    "chelsea": data.chelsea,       # fur texture, muted palette
    "rocket": data.rocket,         # sky, hard edges, wide dynamic range
    "page": lambda: np.dstack([data.page()] * 3),   # monochrome scan
}

PROFILES = {
    "dim_noisy":  {"blur": 1.4, "noise": 0.018, "downscale": 2,
                   "exposure": 0.5, "flatten": 0.6, "cast": (1.0, 0.92, 0.78)},
    "soft_hazy":  {"blur": 1.8, "noise": 0.004, "downscale": 3,
                   "exposure": 0.85, "flatten": 0.75, "cast": (0.85, 0.95, 1.15)},
    "mild":       {"blur": 0.8, "noise": 0.006, "downscale": 2,
                   "exposure": 0.9, "flatten": 0.9, "cast": (1.0, 0.98, 0.95)},
    # The control. Input == original, so every non-zero number is damage.
    "already_ok": {"blur": 0.0, "noise": 0.0, "downscale": 1,
                   "exposure": 1.0, "flatten": 1.0, "cast": (1.0, 1.0, 1.0)},
}

KEYS = ["shadow_clip_vs_orig", "highlight_clip_vs_orig", "sat_delta_vs_orig",
        "halo", "local_std_vs_orig", "skin_shift", "rmse_vs_orig"]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def clip_fracs(img: np.ndarray) -> tuple[float, float]:
    y = luma(img)
    return float(np.mean(y < 0.02)), float(np.mean(y > 0.98))


def mean_chroma(img: np.ndarray) -> float:
    lab = rgb_to_lab(img)
    return float(np.mean(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)))


def halo_energy(before: np.ndarray, after: np.ndarray) -> float:
    """How far the result escapes the input's local tonal range, along edges.

    That excursion is what a halo physically *is*, so measuring it directly
    beats any proxy: a bright rim outside the range the neighbourhood ever had
    is visible as a halo and nothing else.
    """
    b = luma(after)
    a = cv2.resize(luma(before), (b.shape[1], b.shape[0]),
                   interpolation=cv2.INTER_LANCZOS4)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    lo, hi = cv2.erode(a, k), cv2.dilate(a, k)
    edges = cv2.dilate(cv2.Canny((a * 255).astype(np.uint8), 60, 160), k) > 0
    if edges.sum() < 32:
        return 0.0
    over = np.maximum(b - hi, 0) + np.maximum(lo - b, 0)
    return float(np.mean(over[edges]))


def local_std(img: np.ndarray, k: int = 9) -> float:
    y = luma(img)
    m = cv2.blur(y, (k, k))
    return float(np.mean(np.sqrt(np.maximum(cv2.blur(y * y, (k, k)) - m * m, 0))))


def skin_hue(img: np.ndarray) -> float:
    lab = rgb_to_lab(img)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    hue = np.arctan2(b, a)
    m = (chroma > 8) & (chroma < 55) & (hue > 0.3) & (hue < 1.1)
    return float(np.mean(hue[m])) if m.sum() > 200 else float("nan")


def evaluate(before: np.ndarray, after: np.ndarray, truth: np.ndarray) -> dict:
    t = cv2.resize(truth, (after.shape[1], after.shape[0]),
                   interpolation=cv2.INTER_AREA)
    st, ht = clip_fracs(t)
    sa, ha = clip_fracs(after)
    return {
        "shadow_clip_vs_orig": sa - st,
        "highlight_clip_vs_orig": ha - ht,
        "sat_delta_vs_orig": mean_chroma(after) - mean_chroma(t),
        "halo": halo_energy(before, after),
        "local_std_vs_orig": local_std(after) / max(local_std(t), 1e-6),
        "skin_shift": abs(skin_hue(after) - skin_hue(t)),
        "rmse_vs_orig": float(np.sqrt(np.mean((after - t) ** 2))),
    }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def _contact_sheet(panels: list[np.ndarray], width: int = 2200) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(1, 1, 1))
              for p in panels]
    half = max(1, len(panels) // 2)
    rows = [np.concatenate(panels[:half], axis=1),
            np.concatenate(panels[half:], axis=1)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(1, 1, 1)) for r in rows]
    sheet = np.concatenate(rows, axis=0)
    f = min(1.0, width / sheet.shape[1])
    return cv2.resize(sheet, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def main(tag: str = "run", scale: float = 2.0, preset: str = "natural",
         out_dir: str = "benchmarks/out") -> None:
    rows, panels = [], []

    for sname, loader in SCENES.items():
        truth = loader().astype(np.float32) / 255.0
        for pname, profile in PROFILES.items():
            degraded = degrade(truth, **profile)
            result = pl.enhance(degraded, pl.preset(preset, scale=scale))
            row = evaluate(degraded, result.image, truth)
            row["scene"], row["profile"] = sname, pname
            rows.append(row)
            if pname in ("dim_noisy", "soft_hazy"):
                panels.append(pl.side_by_side(degraded, result.image))

    print(f"\n===== {tag} (preset={preset}, scale={scale}) =====")
    print(f"{'scene/profile':26s}" + "".join(f"{k[:11]:>13s}" for k in KEYS))
    for r in rows:
        print(f"{r['scene'] + '/' + r['profile']:26s}"
              + "".join(f"{r.get(k, float('nan')):13.4f}" for k in KEYS))

    def agg(fn):
        return "".join(f"{fn([r.get(k, np.nan) for r in rows]):13.4f}" for k in KEYS)

    print(f"{'MEAN':26s}" + agg(np.nanmean))
    print(f"{'WORST':26s}"
          + "".join(f"{np.nanmax([abs(r.get(k, np.nan)) for r in rows]):13.4f}"
                    for k in KEYS))
    print("\nideal: ratios 1.0, differences 0.0. The already_ok rows matter "
          "most -- there is nothing to fix there, so any deviation is damage.")

    path = Path(out_dir) / f"sheet_{tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.save(_contact_sheet(panels), str(path))
    print(f"-> {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag", nargs="?", default="run",
                    help="label for this run; names the output sheet")
    ap.add_argument("-s", "--scale", type=float, default=2.0)
    ap.add_argument("-p", "--preset", default="natural", choices=sorted(pl.PRESETS))
    ap.add_argument("-o", "--out-dir", default="benchmarks/out")
    args = ap.parse_args()
    main(args.tag, args.scale, args.preset, args.out_dir)
