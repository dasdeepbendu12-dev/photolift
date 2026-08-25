# Changelog

All notable changes to photolift are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.2.0] — 2026-08-25

The naturalness release. Version 0.1 over-processed: on an image that needed
nothing it lifted genuine blacks, dragged deliberately low-key frames to
mid-grey, and neutralised intentional colour. The default now aims to make a
photograph look like itself.

This is a behavioural change, not an API break. Existing calls keep working;
they produce more restrained output. The previous look is available as
`preset("punchy")` or `intensity=2.0`.

### Added

- **`intensity`** config knob (default `1.0`) scaling the authority of every
  automatic *stylistic* correction. Denoise and deconvolve are deliberately
  excluded — noise and blur are measured physical defects, not matters of
  taste. Exposed on the CLI as `-i/--intensity`.
- **`natural_guard`** (default `True`): after the pipeline runs, the result is
  measured against the input, and if it crushed shadows, blew highlights, or
  inflated saturation past tolerance, the whole thing is redone with less
  authority. Runs at most twice and keeps the retry only if it is genuinely
  better. Disabled by default in `punchy`, `aggressive` and `document`, where
  overshooting is the point. CLI: `--no-guard`.
- **`ops.flat_denoise`** and the matching `flat_denoise` config knob: a
  post-upscale stage that smooths only regions containing no signal above the
  noise floor. The contrast stages amplify residual grain most visibly in
  smooth areas, and denoising harder up front would cost texture everywhere to
  fix a problem that only shows up in flat regions. CLI: `--flat-denoise`.
- **`vivid`** (intensity 1.6) and **`punchy`** (intensity 2.0) presets.
- `sharpen_threshold` is now auto-set from measured noise, so sharpening skips
  flat areas on a grainy source instead of putting back the grain that
  denoising just removed.
- Twelve naturalness regression tests locking in the behaviour above.

### Changed

- **Auto-strength is severity-proportional.** Every knob's cap now rises only
  once the measurement sits far enough outside normal that the defect cannot
  plausibly be an artistic choice. A flat cap tuned to protect good
  photographs also refuses to rescue bad ones, and vice versa.
- **Deep blacks are treated as intent.** The shadow lift is gated on the frame
  being dark *overall*, not merely on the presence of clipped pixels — which
  describes every studio portrait ever taken.
- **Colourfulness is now tracked separately from saturation.** A greyscale
  scan with an age-tint has high saturation and near-zero colourfulness. An
  image with no colour *variety* cannot have an intentional colour grade, so
  its cast is removed completely and it receives no saturation push at all.
- Exposure, white balance and levels all close part of the gap rather than all
  of it, with wider deadbands before engaging.
- Saturation targets an *ordinary* palette rather than a vivid one. With no
  reference there is no way to know whether a photo was richly coloured or
  muted to begin with; genuinely vivid images come back slightly under, which
  is correctable with `vibrance=` or `intensity=`.
- Denoise is gentler on luma (`h` 12 → 9) and harder on chroma (20 → 24).
- The test suite is split by topic (`test_analysis.py`, `test_ops.py`,
  `test_upscale.py`, `test_pipeline.py`, `test_cli.py`,
  `test_naturalness.py`).

### Fixed

- **Denoise re-injected the noise it had just removed.** `preserve_detail` was
  a fixed 0.5, but the residual it blends back contains the noise *and* the
  texture, so at high strength — exactly when there was a lot of noise — half
  of it came straight back. It now tapers with strength.
- **`intensity > 1` multiplied past the per-knob caps.** Caps were applied
  before scaling, producing values like `white_balance=1.57`, which does not
  remove a colour cast but inverts it. Caps are now re-applied after scaling.
- **The naturalness guard over-damped**, landing `vivid` *weaker* than
  `natural` — a worse result than the one it was correcting. The damping curve
  is gentler and floored well above zero.
- **`flat_denoise` measured flatness using local variance that included the
  noise it was removing**, so its mask collapsed to zero and the operator
  silently did nothing precisely when it was needed. The threshold is now set
  relative to the measured noise sigma.
- `resize_only` did not pin the new `flat_denoise` to zero, so it no longer
  changed only the size.
- Dehaze's transmission floor raised from 0.10 to 0.30. Dividing by `t`
  multiplies everything in a region by `1/t`, including grain, so a floor of
  0.10 was a 10x noise gain in exactly the dark areas where noise is worst.
- Denoise now runs *before* dehaze and levels rather than after. Both amplify
  shadow noise into blotchy chroma mottling that no later stage can remove.

## [0.1.0] — 2026-08-25

Initial release.

### Added

- Automatic enhancement pipeline: white balance, exposure, denoise, shadows and
  highlights, dehaze, levels, deconvolution, local contrast, tone curve,
  clarity, vibrance, sharpening, grain.
- Upscaling backends: `lanczos`, `cubic`, `iterative` (Lanczos plus iterative
  back-projection, the default), `dnn` (OpenCV `dnn_superres`) and
  `realesrgan` (optional).
- `analyze()` and `ImageStats`: noise-aware sharpness, blur radius, noise
  sigma, colour cast, exposure, dynamic range, entropy, colourfulness.
- Presets: `natural`, `portrait`, `landscape`, `document`, `lowlight`,
  `aggressive`, `resize_only`.
- `photolift` command line interface with batch mode, `--inspect` and
  `--compare`.
- Before/after helpers: `side_by_side`, `split_view`, `zoom_strip`.

### Fixed

Bugs caught during development, listed because each was invisible in casual use
and each would have silently degraded every output:

- `replace_luma` wrote Rec.709 luma into a CIE L\* channel. L\* is perceptual
  lightness on a 0–100 scale, not luma; the mismatch darkened midtones by
  roughly 20% through *every* luminance-based operator.
- White balance normalised its gains to their maximum, so channels only ever
  moved down and correcting a cast always darkened the image.
- The sharpness metric was clipped rather than saturating, pinning at 1.0 for
  any detailed image and so unable to distinguish sharp from sharper.
- The blur estimator compared an image's energy against blurred copies of
  *itself*, which is circular and always reported zero blur.
- Iterative back-projection diverged, producing higher reprojection error than
  plain Lanczos, due to a decaying step size combined with a correction blur.
- `--backends` required an image argument despite being informational.

[Unreleased]: https://github.com/dasdeepbendu12-dev/photolift/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dasdeepbendu12-dev/photolift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dasdeepbendu12-dev/photolift/releases/tag/v0.1.0
