"""Install script for PyFibers."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _clean_mod_dir(mod_dir: str) -> None:
    """Remove generated C/C++ build intermediates from ``mod_dir``."""  # noqa: DAR101
    removable_extensions = (".c", ".cpp", ".o")
    for name in os.listdir(mod_dir):
        path = os.path.join(mod_dir, name)
        if not os.path.isfile(path):
            continue

        _, extension = os.path.splitext(name)
        if extension.lower() in removable_extensions:
            os.remove(path)


def running_compile() -> bool:
    """Return whether current process was launched via ``pyfibers_compile``."""  # noqa: DAR201
    argv0 = os.path.basename(sys.argv[0]).lower()
    return "pyfibers_compile" in argv0


def _has_generated_c_files(mod_dir: str) -> bool:
    """Return whether generated ``.c`` files are present in ``mod_dir``."""  # noqa: DAR101, DAR201
    return any(os.path.splitext(name)[1].lower() == ".c" for name in os.listdir(mod_dir))


def main(argv: list[str] | None = None) -> None:
    """Compile NEURON MOD files. # noqa: DAR101.

    :raises RuntimeError: If nrnivmodl is not found or fails.
    """
    parser = argparse.ArgumentParser(description="Compile PyFibers NEURON mechanism (.mod) files.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Force removal of generated .c, .cpp, and .o files before compiling.",
    )
    args = parser.parse_args(argv)

    mod_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MOD")

    if args.clean or _has_generated_c_files(mod_dir):
        _clean_mod_dir(mod_dir)

    # Change to MOD directory
    os.chdir(mod_dir)

    # Check for nrnivmodl
    if not shutil.which("nrnivmodl"):
        raise RuntimeError(
            "nrnivmodl not found. Please install NEURON and add to PATH, or your NEURON version may be " "too old."
        )
    # Run nrnivmodl
    try:
        subprocess.check_call(shutil.which('nrnivmodl'))
    except subprocess.CalledProcessError:
        raise RuntimeError("NEURON compilation command (nrnivmodl) failed. Please check the output for errors.")

    # check that post compilation files based on OS exist
    if os.name == 'nt':
        file_to_check = 'nrnmech.dll'
    elif os.uname().machine == 'arm64':  # type: ignore
        file_to_check = 'arm64/special'
    else:
        file_to_check = 'x86_64/special'

    full_file_to_check = os.path.join(os.getcwd(), file_to_check)
    if not os.path.exists(full_file_to_check):
        raise RuntimeError(f"{file_to_check} not found, install failed. Please check the output for errors.")


if __name__ == "__main__":
    main()
