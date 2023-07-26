"""Packages the analyzer and the gui into an application.
Librosa 0.9.2 has to be used, since pyinstaller cant package >=0.10.0.
See https://github.com/librosa/librosa/issues/1705.
"""
import pathlib
import sys
import zipfile

import PyInstaller.__main__


def build(app_name: str, create_zip=False):
    PyInstaller.__main__.run(
        [
            "--clean",
            "--noconfirm",
            f"{app_name}-full.spec",
        ]
    )

    if create_zip:
        print("Creating zip file.")
        dist_dir = pathlib.Path(sys.argv[0]).parent / "dist"
        analyzer_dir = dist_dir / app_name

        with zipfile.ZipFile(
            dist_dir / f"{app_name}.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for entry in analyzer_dir.rglob("*"):
                archive.write(entry, entry.relative_to(analyzer_dir.parent))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="parser for creating build.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="BirdNET-Analyzer",
        help="Represents the name of the app.",
    )
    parser.add_argument(
        "-z",
        "--zip",
        action="store_true",
    )

    args, _ = parser.parse_known_args()

    build(args.name, create_zip=args.zip)
