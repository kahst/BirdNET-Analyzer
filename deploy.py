"""Module for deployment as an application.

Raises:
    ValueError: When the specified os is not supported.
"""
import PyInstaller.__main__


def build(target_os):
    """Uses PyInstaller to build a BirdNET application.

    Args:
        target_os (str): The targeted operating system.

    Raises:
        ValueError: Is raised if the specified operating system is not supported.
    """
    if target_os not in ("win", "mac"):
        raise ValueError(f"OS {target_os} is not supported use win or mac.")

    PyInstaller.__main__.run(["--clean", "--noconfirm", f"BirdNET-Analyzer-{target_os}.spec"])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bundles BirdNET into an application.")
    parser.add_argument("target_os", choices=["win", "mac"], help="Choose the operating for which the application should be build.")
    args, _ = parser.parse_known_args()

    build(args.target_os)
