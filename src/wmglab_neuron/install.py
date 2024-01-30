"""Install script for wmglab_neuron."""

from __future__ import annotations

import os
import shutil
import subprocess


def main() -> None:
    """Compile NEURON MOD files.

    :raises RuntimeError: If nrnivmodl is not found or fails.
    """
    # Change to MOD directory
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MOD"))
    # Copy all txt files and rename to .mod
    # remove all .mod and .c files
    for file in os.listdir():
        if file.endswith(".mod") or file.endswith(".c"):
            os.remove(file)

    for file in os.listdir():
        if file.endswith(".txt"):
            os.rename(file, file[:-4] + ".mod")
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
    file_to_check = 'nrnmech.dll' if os.name == 'nt' else 'x86_64/special'
    full_file_to_check = os.path.join(os.getcwd(), file_to_check)
    if not os.path.exists(full_file_to_check):
        raise RuntimeError(f"{file_to_check} not found, install failed. Please check the output for errors.")


if __name__ == "__main__":
    main()
