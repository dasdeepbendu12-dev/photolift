"""Command line interface: ``photolift``."""

from __future__ import annotations

import argparse
import glob
import os
import sys

from . import __version__
from .analysis import analyze
from .compare import side_by_side
from .imageio import load, save
from .pipeline import PRESETS, EnhanceConfig, Enhancer, preset
from .upscale import BACKENDS, available_backends

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photolift",
        description="Upscale and enhance photographs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  photolift photo.jpg -o big.png --scale 4\n"
            "  photolift shots/ -o out/ --preset lowlight --scale 2\n"
            "  photolift scan.jpg -o clean.png --preset document --compare cmp.png\n"
            "  photolift photo.jpg --inspect\n"
            "  photolift flat.jpg -o out.png --preset vivid\n"
        ),
    )
    # Not "+": --backends and --version are informational and take no input.
    p.add_argument("input", nargs="*", help="image file(s), directory, or glob")
    p.add_argument("-o", "--output", help="output file or directory")
    p.add_argument("-s", "--scale", type=float, default=2.0,
                   help="resolution multiplier (default: 2)")
    p.add_argument("-p", "--preset", choices=sorted(PRESETS), default="natural")
    p.add_argument("-b", "--backend", choices=BACKENDS, default="auto")
    p.add_argument("--model", help="path to a dnn_superres .pb model")
    p.add_argument("-i", "--intensity", type=float, metavar="0..2",
                   help="how hard the automatic corrections push "
                        "(1.0 = natural, the default; 1.6 = vivid; 2.0 = punchy)")
    p.add_argument("--no-guard", action="store_true",
                   help="disable the naturalness guard that backs off when the "
                        "result crushes shadows, blows highlights or oversaturates")

    g = p.add_argument_group("manual overrides (any omitted knob stays automatic)")
    g.add_argument("--white-balance", type=float, metavar="0..1")
    g.add_argument("--exposure", type=float, metavar="0..1")
    g.add_argument("--shadows", type=float, metavar="0..1")
    g.add_argument("--highlights", type=float, metavar="0..1")
    g.add_argument("--dehaze", type=float, metavar="0..1")
    g.add_argument("--levels", type=float, metavar="0..1")
    g.add_argument("--local-contrast", type=float, metavar="0..4")
    g.add_argument("--contrast", type=float, metavar="-1..1")
    g.add_argument("--clarity", type=float, metavar="0..1")
    g.add_argument("--denoise", type=float, metavar="0..1")
    g.add_argument("--flat-denoise", type=float, metavar="0..1",
                   help="post-upscale cleanup of grain in smooth areas")
    g.add_argument("--deconvolve", type=float, metavar="sigma-px")
    g.add_argument("--sharpen", type=float, metavar="0..2")
    g.add_argument("--vibrance", type=float, metavar="-1..1")
    g.add_argument("--grain", type=float, metavar="0..0.05")
    g.add_argument("--no-auto", action="store_true",
                   help="disable automatic strengths entirely")

    o = p.add_argument_group("output")
    o.add_argument("-q", "--quality", type=int, default=95, help="JPEG/WebP quality")
    o.add_argument("--bit-depth", type=int, choices=(8, 16))
    o.add_argument("--ext", help="force output extension for batch mode, e.g. .png")
    o.add_argument("--suffix", default="_enhanced", help="batch filename suffix")
    o.add_argument("--compare", metavar="PATH",
                   help="also write a before/after sheet (single input only)")
    o.add_argument("--max-megapixels", type=float, default=80.0)

    m = p.add_argument_group("modes")
    m.add_argument("--inspect", action="store_true",
                   help="measure and report, change nothing")
    m.add_argument("--backends", action="store_true",
                   help="list upscaling backends available here, then exit")
    m.add_argument("-v", "--verbose", action="store_true")
    m.add_argument("--version", action="version", version=f"photolift {__version__}")
    return p


def _expand(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            for name in sorted(os.listdir(pattern)):
                if name.lower().endswith(_IMAGE_EXTS):
                    files.append(os.path.join(pattern, name))
        elif any(ch in pattern for ch in "*?["):
            files.extend(sorted(glob.glob(pattern)))
        else:
            files.append(pattern)
    return files


def _config_from_args(args: argparse.Namespace) -> EnhanceConfig:
    cfg = preset(args.preset)
    return cfg.merged(
        auto=not args.no_auto,
        intensity=args.intensity,
        natural_guard=False if args.no_guard else None,
        scale=args.scale,
        backend=args.backend,
        model_path=args.model,
        max_megapixels=args.max_megapixels,
        white_balance=args.white_balance,
        exposure=args.exposure,
        shadows=args.shadows,
        highlights=args.highlights,
        dehaze=args.dehaze,
        levels=args.levels,
        local_contrast=args.local_contrast,
        contrast=args.contrast,
        clarity=args.clarity,
        denoise=args.denoise,
        flat_denoise=args.flat_denoise,
        deconvolve=args.deconvolve,
        sharpen=args.sharpen,
        vibrance=args.vibrance,
        grain=args.grain,
        quality=args.quality,
        bit_depth=args.bit_depth,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.backends:
        for name, ok in available_backends(args.model).items():
            print(f"  {'yes' if ok else ' no'}  {name}")
        return 0

    files = _expand(args.input)
    if not files:
        print("photolift: no input images matched", file=sys.stderr)
        return 2

    if args.inspect:
        for path in files:
            print(f"\n{path}")
            print("  " + analyze(load(path)).summary().replace("\n", "\n  "))
        return 0

    if not args.output:
        print("photolift: -o/--output is required", file=sys.stderr)
        return 2

    cfg = _config_from_args(args)
    log = (lambda m: print(f"    {m}")) if args.verbose else None
    enhancer = Enhancer(cfg, progress=log)

    batch = len(files) > 1 or os.path.isdir(args.output) or args.output.endswith(os.sep)
    if batch:
        os.makedirs(args.output, exist_ok=True)

    failures = 0
    for path in files:
        if batch:
            stem, src_ext = os.path.splitext(os.path.basename(path))
            dst = os.path.join(args.output, f"{stem}{args.suffix}{args.ext or src_ext}")
        else:
            dst = args.output

        try:
            result = enhancer.enhance(path)
            result.save(dst, quality=cfg.quality, bit_depth=cfg.bit_depth)
        except Exception as exc:
            failures += 1
            print(f"  FAILED {path}: {exc}", file=sys.stderr)
            continue

        b, a = result.stats_before, result.stats_after
        print(f"{path} -> {dst}  "
              f"[{b.width}x{b.height} -> {a.width}x{a.height}, {result.elapsed:.2f}s]")
        if args.verbose:
            print("  " + result.report().replace("\n", "\n  "))
        elif b.flags:
            print(f"  fixed: {', '.join(b.flags)}")

        if args.compare and not batch:
            save(side_by_side(load(path), result.image), args.compare)
            print(f"  comparison -> {args.compare}")

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
