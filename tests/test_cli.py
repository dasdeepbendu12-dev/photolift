"""Tests for the ``photolift`` command.

The CLI is the interface most users will touch first, so its failure modes get
tested as carefully as the library: informational flags must not demand an
input file, a missing output must be a clear error rather than a traceback,
and a bad path must not return success.
"""

from __future__ import annotations

import os

import photolift as pl
from photolift.cli import main as cli_main


def test_cli_inspect(tmp_path, degraded, capsys):
    src = tmp_path / "in.png"
    pl.save(degraded, src)
    assert cli_main([str(src), "--inspect"]) == 0
    assert "brightness" in capsys.readouterr().out


def test_cli_enhance_writes_file(tmp_path, degraded):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    pl.save(degraded, src)
    assert cli_main([str(src), "-o", str(dst), "--scale", "2"]) == 0
    assert dst.exists()


def test_cli_batch_directory(tmp_path, degraded):
    src = tmp_path / "src"
    src.mkdir()
    pl.save(degraded, src / "a.png")
    pl.save(degraded, src / "b.png")
    out = tmp_path / "out"
    assert cli_main([str(src), "-o", str(out), "--preset", "landscape"]) == 0
    assert len(list(out.iterdir())) == 2


def test_cli_comparison_sheet(tmp_path, degraded):
    src, dst, cmp_path = tmp_path / "i.png", tmp_path / "o.png", tmp_path / "c.png"
    pl.save(degraded, src)
    assert cli_main([str(src), "-o", str(dst), "--compare", str(cmp_path)]) == 0
    assert cmp_path.exists()


def test_cli_requires_output(tmp_path, degraded, capsys):
    src = tmp_path / "in.png"
    pl.save(degraded, src)
    assert cli_main([str(src)]) == 2


def test_cli_no_match_returns_error(capsys):
    assert cli_main([os.path.join("definitely", "missing", "*.png")]) == 2


def test_cli_backends_needs_no_input(capsys):
    """--backends is informational; requiring an image for it is a bug."""
    assert cli_main(["--backends"]) == 0
    assert "lanczos" in capsys.readouterr().out


def test_cli_bare_invocation_is_an_error(capsys):
    assert cli_main([]) == 2
