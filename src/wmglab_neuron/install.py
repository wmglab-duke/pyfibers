"""Install script for wmglab_neuron."""
import os
import shutil
import subprocess


def main():
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
        subprocess.check_call(["nrnivmodl"])
    except subprocess.CalledProcessError:
        raise RuntimeError("NEURON compilation command (nrnivmodl) failed. Please check the output for errors.")


if __name__ == "__main__":
    main()
