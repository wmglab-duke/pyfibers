"""Tests for pyfibers.compile helpers (does not run nrnivmodl).

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import Mock

import pytest

from pyfibers.compile import _clean_mod_dir, _has_generated_c_files, main, running_compile


def test_running_compile_true_for_argv0(monkeypatch):
    monkeypatch.setattr(sys, "argv", [r"C:\env\Scripts\pyfibers_compile.exe"])
    assert running_compile() is True


def test_running_compile_false_otherwise(monkeypatch):
    monkeypatch.setattr(sys, "argv", [sys.executable, "-m", "pytest"])
    assert running_compile() is False


def test_clean_mod_dir_removes_c_cpp_o(tmp_path):
    keep = tmp_path / "keep.mod"
    keep.write_text("dummy")
    (tmp_path / "foo.c").write_text("c")
    (tmp_path / "foo.cpp").write_text("cpp")
    (tmp_path / "foo.o").write_text("o")
    (tmp_path / "subdir").mkdir()
    _clean_mod_dir(str(tmp_path))
    assert keep.exists()
    assert not (tmp_path / "foo.c").exists()
    assert not (tmp_path / "foo.cpp").exists()
    assert not (tmp_path / "foo.o").exists()


def test_has_generated_c_files(tmp_path):
    assert _has_generated_c_files(str(tmp_path)) is False
    (tmp_path / "foo.c").write_text("c")
    assert _has_generated_c_files(str(tmp_path)) is True


def test_main_nrnivmodl_missing(monkeypatch):
    import pyfibers.compile as compile_mod

    monkeypatch.setattr(compile_mod.os, "chdir", lambda _path: None)
    monkeypatch.setattr(compile_mod, "_has_generated_c_files", lambda _path: False)
    monkeypatch.setattr(compile_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="nrnivmodl not found"):
        main([])


def test_main_nrnivmodl_fails(monkeypatch):
    import pyfibers.compile as compile_mod

    monkeypatch.setattr(compile_mod.os, "chdir", lambda _path: None)
    monkeypatch.setattr(compile_mod, "_has_generated_c_files", lambda _path: False)
    monkeypatch.setattr(compile_mod.shutil, "which", lambda _name: "nrnivmodl")
    monkeypatch.setattr(
        compile_mod.subprocess,
        "check_call",
        Mock(side_effect=subprocess.CalledProcessError(1, "nrnivmodl")),
    )
    with pytest.raises(RuntimeError, match="nrnivmodl\\) failed"):
        main([])


def test_main_missing_output_file(monkeypatch):
    import pyfibers.compile as compile_mod

    monkeypatch.setattr(compile_mod.os, "chdir", lambda _path: None)
    monkeypatch.setattr(compile_mod, "_has_generated_c_files", lambda _path: False)
    monkeypatch.setattr(compile_mod.shutil, "which", lambda _name: "nrnivmodl")
    monkeypatch.setattr(compile_mod.subprocess, "check_call", lambda _cmd: 0)
    monkeypatch.setattr(compile_mod.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(compile_mod.os, "name", "nt")
    with pytest.raises(RuntimeError, match="nrnmech.dll not found"):
        main([])


def test_main_clean_flag_calls_clean(monkeypatch):
    import pyfibers.compile as compile_mod

    cleaned = {}
    monkeypatch.setattr(compile_mod, "_clean_mod_dir", lambda path: cleaned.setdefault("path", path))
    monkeypatch.setattr(compile_mod, "_has_generated_c_files", lambda _path: False)
    monkeypatch.setattr(compile_mod.os, "chdir", lambda _path: None)
    monkeypatch.setattr(compile_mod.shutil, "which", lambda _name: "nrnivmodl")
    monkeypatch.setattr(compile_mod.subprocess, "check_call", lambda _cmd: 0)
    monkeypatch.setattr(compile_mod.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(compile_mod.os, "name", "nt")
    main(["--clean"])
    assert "path" in cleaned
