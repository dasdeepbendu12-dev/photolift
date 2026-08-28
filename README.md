# photolift

[![CI](https://github.com/dasdeepbendu12-dev/photolift/actions/workflows/ci.yml/badge.svg)](https://github.com/dasdeepbendu12-dev/photolift/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/dasdeepbendu12-dev/photolift?label=release&color=blue)](https://github.com/dasdeepbendu12-dev/photolift/releases)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Turn a small, soft, badly-lit photo into a large, clean one.

```bash
pip install git+https://github.com/dasdeepbendu12-dev/photolift
photolift dim_phone_photo.jpg -o fixed.png --scale 4
```

```python
import photolift as pl

result = pl.enhance("dim_phone_photo.jpg", scale=2)
result.save("fixed.png")
print(result.report())
```

---

## The one thing to know

The default output is tuned to look like **a good photograph**, not like an
enhanced one. Every automatic correction has a wide deadband before it engages
and a cap well short of a full correction, because a partial fix reads as a
better photo while a complete fix reads as a processed one. A photo that is
already good comes back essentially unchanged.

If you want the punchier look, ask for it:

```bash
photolift photo.jpg -o out.png --preset vivid      # more authority
photolift photo.jpg -o out.png --intensity 1.4     # or dial it yourself
```

---

## What this can and cannot do

Read this part before anything else, because it determines whether photolift
is the right tool for your problem.

The package does **two separable things** and keeps them separate on purpose:

**Enhancement** fixes what your pixels already contain but present badly —
wrong exposure, a colour cast, flat contrast, haze, sensor noise, soft focus.
This is real recovery. The information is in the file; it is just encoded
poorly. photolift will reliably improve these.

**Upscaling** adds pixels. The default backends add *no information* — they
interpolate, intelligently, and the result is a larger file that looks about
as detailed as the original did. The optional learned backends (`dnn`,
`realesrgan`) do better, but they achieve it by **inventing plausible detail**
based on what they were trained on. That is often what you want. It is not the
same as recovering what the camera captured, and on faces, text and licence
plates it will confidently produce things that were never there.

Nothing here reconstructs detail your sensor never recorded. Any tool that
claims otherwise is either using a learned model and not telling you, or lying.

---

## Install

Not on PyPI yet, so install from source:

```bash
# classical backends, no GPU, no model downloads
pip install git+https://github.com/dasdeepbendu12-dev/photolift

# adds the optional Real-ESRGAN backend (pulls in torch, ~2GB)
pip install "photolift[ai] @ git+https://github.com/dasdeepbendu12-dev/photolift"
```

Requires Python 3.9-3.13. The core dependency is `opencv-contrib-python`,
which works on both 4.x and 5.x.

On Linux, OpenCV links against GL and glib even when used headlessly. If the
import fails with `libGL.so.1: cannot open shared object file`, install the
system packages rather than reinstalling OpenCV:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

---

## Command line

```bash
# basics
photolift photo.jpg -o big.png --scale 4

# a whole folder
photolift shots/ -o enhanced/ --preset lowlight --scale 2

# see what it thinks is wrong, change nothing
photolift photo.jpg --inspect

# get a before/after sheet so you can actually judge the result
photolift scan.jpg -o clean.png --preset document --compare check.png

# show your work
photolift photo.jpg -o out.png --verbose

# override one knob, leave the rest automatic
photolift photo.jpg -o out.png --sharpen 0.3 --denoise 0

# which upscalers can this machine run?
photolift --backends
```

`--inspect` output looks like this:

```
photo.jpg
  1024x768 (0.8MP) | brightness 0.31 | contrast 0.071 | sharpness 0.44
  (blur ~1.10px) | noise 5.2/255 | saturation 0.09 | cast 0.31
  flags: underexposed, flat/low-contrast, soft-focus, noisy, colour-cast
```

Those flags are what the automatic mode acts on.

---

## Python API

### The simple path

```python
import photolift as pl

result = pl.enhance("photo.jpg", scale=2)
result.save("out.png")

result.stats_before.summary()   # what was wrong
result.stats_after.summary()    # what it looks like now
result.steps                    # exactly what ran, and how hard
```

### Presets

```python
pl.enhance("portrait.jpg", pl.preset("portrait", scale=2))
```

| preset | for | character |
|---|---|---|
| `natural` | anything | **default.** Fixes what is measurably wrong, then stops |
| `vivid` | flat sources | same decisions, more authority (`intensity` 1.6) |
| `punchy` | when you want it obvious | deliberately processed (`intensity` 2.0) |
| `portrait` | people | gentle sharpening, protected skin tones, no crunch |
| `landscape` | scenery | clarity, dehaze, richer colour |
| `document` | scans, text, whiteboards | hard levels, maximum acutance, no grain |
| `lowlight` | night, high-ISO | heavy denoise, lifted shadows, restrained colour |
| `aggressive` | genuinely bad sources | will look processed — that is the deal |
| `resize_only` | a control | changes resolution and nothing else |

### Intensity

One knob scales the authority of every automatic *stylistic* correction:

```python
pl.enhance("photo.jpg", intensity=0.6)   # near-invisible touch-up
pl.enhance("photo.jpg", intensity=1.0)   # default: looks like a photograph
pl.enhance("photo.jpg", intensity=1.8)   # obviously enhanced
```

Denoising and deconvolution are deliberately *not* scaled by it. Noise and
blur are measured physical defects, not matters of taste.

### The naturalness guard

After the pipeline runs, the result is measured against the input. If it
crushed shadows, blew highlights, or inflated saturation past tolerance, the
whole thing is redone with less authority. It runs at most twice, only kicks in
when something actually went too far, and keeps the retry only if the retry is
genuinely better.

This exists because auto-fill guesses strengths from the *input*, and ten
stages then compose in ways no per-stage guess fully predicts. Predicting the
final clipping from input statistics is not reliable; measuring the output is.

```python
pl.enhance("photo.jpg", natural_guard=False)   # off, if you want the raw result
```

It is off by default in `punchy`, `aggressive` and `document`, where
overshooting is the point.

### Automatic, with exceptions

Every knob is `None` by default, meaning "decide from the image". Pin the ones
you care about; the rest stay automatic. This is the intended way to use it.

```python
result = pl.enhance(
    "photo.jpg",
    scale=3,
    sharpen=0.4,      # I want this exact amount
    denoise=0,        # and never touch my grain
)                     # everything else: measured and decided
```

Turn the whole thing off with `auto=False` and nothing happens unless you
asked for it explicitly:

```python
cfg = pl.EnhanceConfig(auto=False, scale=2, sharpen=0.5)
pl.enhance("photo.jpg", cfg)      # upscale and sharpen. Nothing more.
```

### Batch

```python
pl.enhance_batch(
    glob.glob("raw/*.jpg"),
    "out/",
    pl.preset("lowlight"),
    on_result=lambda src, r: print(src, "ok" if not isinstance(r, Exception) else r),
)
```

One corrupt file does not abort the run.

### Individual operators

Every stage is importable and usable on its own, on any float RGB array:

```python
from photolift import ops

img = pl.load("photo.jpg")
img = ops.white_balance(img, strength=0.8)
img = ops.shadows_highlights(img, shadows=0.4, highlights=0.2)
img = ops.deconvolve(img, sigma=1.1)          # real focus recovery
img = ops.unsharp_mask(img, amount=0.6)
pl.save(img, "out.png")
```

### Judging the result

Never trust an enhancement you have only seen at fit-to-screen.

```python
pl.save(pl.side_by_side(before, after), "sheet.png")
pl.save(pl.split_view(before, after), "wipe.png")
pl.save(pl.zoom_strip(before, after), "pixels.png")   # 1:1 crop, both frames
```

---

## Upscaling backends

| backend | needs | invents detail? | notes |
|---|---|---|---|
| `lanczos` | nothing | no | fast, safe, soft |
| `iterative` | nothing | no | **default.** Lanczos + iterative back-projection |
| `dnn` | a `.pb` model file | yes | OpenCV `dnn_superres` (EDSR / ESPCN / FSRCNN / LapSRN) |
| `realesrgan` | `photolift[ai]` | yes, freely | best on real degraded photos; heavy |

`auto` picks the best one available and falls back quietly.

The `iterative` default deserves a note. After the initial interpolation it
repeatedly downsamples its own output back to the input resolution, measures
what it failed to reproduce, and corrects. It converges on an estimate that is
*consistent* with the original — reprojection error drops by roughly two orders
of magnitude versus plain Lanczos — which reads as noticeably crisper without
any learned prior. It still invents nothing.

For the `dnn` backend, download a model from OpenCV's `dnn_superres` collection
and point at it:

```bash
photolift photo.jpg -o out.png --backend dnn --model EDSR_x4.pb
export PHOTOLIFT_SR_MODEL=/path/to/EDSR_x4.pb   # or set it once
```

---

## What the pipeline actually does, in order

The ordering is the substance of the package, not an implementation detail.

1. **White balance** — shades-of-grey illuminant estimate, luminance-neutral so
   correcting a cast never darkens the frame.
2. **Exposure** — a real multiply in linear light with a soft highlight
   shoulder, not a gamma bend.
3. **Denoise** — non-local means, chroma hit hard and luma gently. Runs *here*,
   before anything that stretches contrast: dehaze divides by a small
   transmission term and levels multiplies the shadows, so running either on a
   noisy frame produces blotchy chroma mottling that no later step can remove.
4. **Shadows / highlights** — edge-aware masks from a guided-filtered
   luminance channel, which is what keeps it from haloing.
5. **Dehaze** — dark channel prior, with a transmission floor set for noise
   control rather than maximum haze removal.
6. **Levels** — percentile black/white point, with a deadband so a full-range
   image is left alone.
7. **Deconvolve** — Richardson–Lucy against a measured Gaussian PSF, at native
   resolution where the blur model is valid. The only step that genuinely
   recovers focus rather than faking it.
8. **Upscale.**
9. **Local contrast, tone curve, clarity, vibrance** — after upscaling, so
   their radii land on the final pixel grid.
10. **Sharpen** — last, at output resolution, from a *fresh* measurement of the
    upscaled image rather than a number computed before the image changed
    underneath it.
11. **Grain** — optional, and less silly than it sounds: a trace of grain hides
    the plastic surface that heavy denoising and upscaling leave behind.

All tone and detail work happens on luminance only, so nothing shifts hue.
Anything physical — exposure, blending, resampling — happens in linear light.

---

## How the automatic mode decides

`photolift.analyze()` measures the image and the strengths follow from the
numbers. The metrics are deliberately noise-aware:

- **sharpness** — Laplacian energy normalised by signal variance, with the
  noise floor subtracted. Noise is high-frequency energy but it is not detail;
  a metric that cannot tell them apart auto-sharpens every grainy photo into
  mush. Saturating rather than clipped, so it can still tell sharp from sharper.
- **blur_sigma** — re-blur by a known amount and measure how much
  high-frequency energy that destroys. Under-reports on very noisy frames,
  which is the safe direction: it only ever decides how hard to deconvolve.
- **noise_sigma** — Immerkær's estimator with a MAD statistic, accurate to
  about 5% in practice.
- **colour cast** — shades-of-grey (Minkowski p=6), not grey-world, which
  embarrasses itself on any image with a large saturated region.

### Trusting the photographer

The auto mode defaults to assuming the photo is the way it is on purpose:

- **a low-key photo stays dark.** Exposure closes part of the gap to the
  target, never all of it.
- **deep blacks are a choice, not a fault.** Shadow clipping alone describes
  every studio portrait ever taken, so the lift only engages when the frame is
  *also* dark overall.
- **a warm photo stays warm.** White balance caps at a partial neutralisation
  -- unless the cast is severe enough that nobody chose it, in which case the
  cap rises.
- **a monochrome image is never tinted.** Colourfulness (the spread of hues
  present) is tracked separately from saturation (how strong they are). A grey
  scan with an age-tint has high saturation and near-zero colourfulness; an
  image with no colour variety cannot have an intentional colour grade, so its
  cast is removed completely and it gets no saturation push at all.

Every cap is *severity-proportional*: it rises only once the measurement sits
far enough outside normal that the defect cannot plausibly be intentional. A
flat cap tuned to protect good photographs also refuses to rescue bad ones.

One honest limitation: with no reference, there is no way to know whether a
photo was richly coloured or muted to begin with, so saturation aims at
*ordinary* and stops. Images that really were vivid come back slightly under.
Correct that with `vibrance=` or `intensity=` -- which is the right way round,
since the alternative is oversaturating everything else.

---

## Repository layout

```
photolift/
├── pyproject.toml            # metadata, dependencies, entry point, tool config
├── README.md
├── CHANGELOG.md
├── LICENSE
├── src/
│   └── photolift/
│       ├── __init__.py       # the public API
│       ├── imageio.py        # load/save, sRGB<->linear, LAB, luma
│       ├── analysis.py       # ImageStats + analyze()
│       ├── ops.py            # the individual operators
│       ├── upscale.py        # resolution backends
│       ├── pipeline.py       # order, auto-strength, presets, guard
│       ├── compare.py        # before/after views
│       └── cli.py            # the `photolift` command
├── tests/
│   ├── conftest.py           # synthetic scenes + the degradation model
│   ├── test_analysis.py
│   ├── test_ops.py
│   ├── test_upscale.py
│   ├── test_pipeline.py
│   ├── test_cli.py
│   └── test_naturalness.py   # regression tests for the default look
├── benchmarks/
│   └── eval_natural.py       # the harness the tuning was done against
└── .github/workflows/ci.yml
```

The `src/` layout is deliberate. With a flat layout, `import photolift` from
the project root picks up the *directory* rather than the installed package, so
tests pass against files that were never packaged and the missing module shows
up after publishing. Under `src/`, nothing imports until it is installed.

---

## Development

```bash
git clone https://github.com/dasdeepbendu12-dev/photolift
cd photolift
pip install -e ".[dev]"
pytest
ruff check src tests benchmarks
```

On Linux, `opencv-contrib-python` links against GL and glib even when used
purely headlessly. If the import fails with `libGL.so.1: cannot open shared
object file`, install the system packages rather than reinstalling OpenCV:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

### Benchmarks

The tuning behind the default look was done against a benchmark, not by eye
alone. It degrades known-good photographs by known amounts and scores the
result **against the original**, because scoring against the degraded input
cannot distinguish "restored the contrast that was lost" from "inflated the
contrast past anything that was there".

```bash
python benchmarks/eval_natural.py baseline     # record where you are
# ...change a constant...
python benchmarks/eval_natural.py candidate    # see what it did
```

Every ratio it reports has an ideal of 1.0 and every difference an ideal of
0.0. Watch the `already_ok` rows hardest: the input there *is* the original,
so any deviation is damage with nothing to justify it.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite builds synthetic scenes, degrades them by known amounts, and asserts
the metrics recover — including that every operator is a true no-op at zero
strength, that back-projection beats interpolation on reprojection error, and
that the sharpness metric is not fooled by noise.

---

## Contributing

Bug reports and pull requests are welcome. Two things make a change easy to
accept:

1. **A test that fails before and passes after.** Every bug in the changelog
   was invisible in casual use and only surfaced because something measured
   it.
2. **Benchmark numbers, if you touch the auto-strength constants.** Run
   `benchmarks/eval_natural.py` before and after and put both tables in the
   PR. A constant that improves one photograph and quietly ruins four others
   is the normal outcome of tuning by eye.

CI runs ruff and the test suite on Python 3.9 through 3.13.

---

## Licence

MIT. See [LICENSE](LICENSE).

Changelog: [CHANGELOG.md](CHANGELOG.md).
