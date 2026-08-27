"""Tests for composition: order, auto-strength, presets, config, batch, IO.

These cover the pipeline as a unit -- what the operators do *together*, which
is where the behaviour that no single operator test can catch lives.
"""

from __future__ import annotations

import numpy as np
import pytest

import photolift as pl


def test_end_to_end_improves_every_targeted_metric(degraded, clean):
    result = pl.enhance(degraded, scale=2)
    b, a = result.stats_before, result.stats_after

    assert a.width == b.width * 2
    assert abs(a.brightness - 0.48) < abs(b.brightness - 0.48)  # exposure fixed
    assert a.contrast > b.contrast                              # contrast fixed
    assert a.wb_cast < b.wb_cast                                # cast removed
    assert a.dynamic_range > b.dynamic_range                    # range restored
    assert a.noise_sigma < b.noise_sigma                        # noise reduced
    assert result.steps and result.elapsed > 0

    # Sharpness is checked on a low-noise degradation, and compared against a
    # plain resize rather than against the input. Both choices are forced by
    # the metric being noise-aware: noise inflates any sharpness measure, so
    # comparing a denoised output to a grainy input asks the wrong question and
    # a heavily grainy source swamps the signal entirely.
    from conftest import degrade
    soft = degrade(clean, blur=1.4, noise=0.0, downscale=2,
                   exposure=0.9, flatten=0.9)
    enhanced = pl.enhance(soft, scale=2).image
    plain = pl.upscale(soft, 2.0, backend="lanczos")
    assert pl.analyze(enhanced).sharpness > pl.analyze(plain).sharpness


def test_auto_leaves_a_good_image_mostly_alone(clean):
    """The real test of an auto mode: it must know when to do nothing."""
    result = pl.enhance(clean, scale=1)
    delta = float(np.abs(result.image - clean).mean())
    assert delta < 0.06


def test_manual_override_wins_over_auto(degraded):
    result = pl.enhance(degraded, scale=1, sharpen=0.0, denoise=0.0)
    names = [name for name, _ in result.steps]
    assert "sharpen" not in names and "denoise" not in names


def test_auto_off_means_nothing_happens(degraded):
    cfg = pl.EnhanceConfig(auto=False, scale=1)
    result = pl.enhance(degraded, cfg)
    assert result.steps == []
    assert np.allclose(result.image, degraded)


@pytest.mark.parametrize("name", sorted(pl.PRESETS))
def test_all_presets_run(degraded, name):
    result = pl.enhance(degraded, pl.preset(name, scale=2))
    assert result.image.shape[:2] == (degraded.shape[0] * 2, degraded.shape[1] * 2)
    assert np.isfinite(result.image).all()


def test_resize_only_preset_changes_nothing_but_size(degraded):
    result = pl.enhance(degraded, pl.preset("resize_only", scale=2))
    assert [n for n, _ in result.steps] == ["upscale"]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        pl.preset("cinematic")


def test_megapixel_guard_clamps_scale(degraded):
    result = pl.enhance(degraded, scale=50, max_megapixels=2.0)
    assert result.stats_after.megapixels <= 2.05


def test_greyscale_input_is_promoted(clean):
    grey = clean.mean(axis=2)
    result = pl.enhance(grey, scale=1)
    assert result.image.ndim == 3


def test_uint8_input_accepted(degraded):
    result = pl.enhance((degraded * 255).astype(np.uint8), scale=1)
    assert result.image.dtype == np.float32


def test_report_is_human_readable(degraded):
    text = pl.enhance(degraded, scale=2).report()
    assert "before:" in text and "after:" in text and "steps" in text


def test_progress_callback_fires(degraded):
    seen: list[str] = []
    pl.Enhancer(pl.EnhanceConfig(scale=2), progress=seen.append).enhance(degraded)
    assert any("upscale" in m for m in seen)


def test_file_roundtrip(tmp_path, degraded):
    src = tmp_path / "in.png"
    pl.save(degraded, src)
    out = tmp_path / "out.png"
    result = pl.enhance_file(str(src), str(out), scale=2)
    assert out.exists() and result.stats_after.width == degraded.shape[1] * 2
    assert pl.load(str(out)).shape[:2] == result.image.shape[:2]


def test_save_respects_bit_depth(tmp_path, clean):
    import cv2
    path = tmp_path / "deep.png"
    pl.save(clean, path, bit_depth=16)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert raw.dtype == np.uint16


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pl.load(tmp_path / "nope.png")


def test_batch_survives_a_broken_file(tmp_path, degraded):
    src = tmp_path / "src"
    src.mkdir()
    pl.save(degraded, src / "a.png")
    pl.save(degraded, src / "b.png")
    (src / "broken.png").write_bytes(b"not an image")

    seen: list[tuple[str, object]] = []
    results = pl.enhance_batch(
        [str(p) for p in sorted(src.iterdir())],
        str(tmp_path / "out"),
        on_result=lambda s, r: seen.append((s, r)),
        scale=2,
    )
    assert len(results) == 2
    assert any(isinstance(r, Exception) for _, r in seen)


def test_comparison_helpers(degraded):
    after = pl.enhance(degraded, scale=2).image
    assert pl.side_by_side(degraded, after).shape[2] == 3
    assert pl.split_view(degraded, after).shape[:2] == after.shape[:2]
    assert pl.zoom_strip(degraded, after).ndim == 3


@pytest.mark.parametrize("labelled", [True, False])
def test_comparison_output_is_float_rgb_in_range(degraded, labelled):
    """Guards the caption bar against OpenCV's drawing-API depth rules.

    ``cv2.putText`` requires an 8-bit image in OpenCV 5; 4.x accepted a float
    one. Drawing on the float canvas therefore passed against a pinned old
    install and raised on every machine with a current OpenCV -- which is the
    failure mode that gets shipped, because it is invisible to the author.
    Asserting the contract of the returned array catches any recurrence.
    """
    after = pl.enhance(degraded, scale=2).image
    sheet = pl.side_by_side(degraded, after, label=labelled)

    assert sheet.dtype == np.float32
    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert sheet.min() >= -1e-6 and sheet.max() <= 1.0 + 1e-6
    # The label bar adds rows above the images; without it, nothing is added.
    assert (sheet.shape[0] > after.shape[0]) == labelled
