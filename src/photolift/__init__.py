"""photolift -- turn a small, soft, badly-lit photo into a large, clean one.

Two things happen here and they are kept deliberately separate:

  *upscaling*    adds pixels. Classical backends add no information; learned
                 backends invent plausible detail. Neither recovers what the
                 sensor never captured, and the package never pretends otherwise.
  *enhancement*  fixes what the pixels already contain but present badly --
                 exposure, colour cast, contrast, haze, noise, focus.

Quick start
-----------
>>> import photolift as pl
>>> result = pl.enhance("dim_phone_photo.jpg", scale=2)
>>> result.save("out.png")
>>> print(result.report())

Everything is automatic by default; every knob can be pinned:

>>> cfg = pl.preset("portrait", scale=3, sharpen=0.4)
>>> pl.enhance_file("in.jpg", "out.jpg", cfg)
"""

from .analysis import ImageStats, analyze
from .compare import side_by_side, split_view, zoom_strip
from .imageio import load, save
from .pipeline import (
    PRESETS,
    EnhanceConfig,
    Enhancer,
    Result,
    enhance,
    enhance_batch,
    enhance_file,
    preset,
)
from .upscale import available_backends, upscale

__version__ = "0.2.1"

__all__ = [
    "__version__",
    # analysis
    "analyze", "ImageStats",
    # pipeline
    "enhance", "enhance_file", "enhance_batch",
    "Enhancer", "EnhanceConfig", "Result", "preset", "PRESETS",
    # pieces
    "upscale", "available_backends", "load", "save",
    "side_by_side", "split_view", "zoom_strip",
    # operator namespace
    "ops",
]

from . import ops  # noqa: E402  (re-exported for direct operator access)
