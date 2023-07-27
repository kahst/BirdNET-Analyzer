"""Packages the analyzer into an application.
Librosa 0.9.2 has to be used, since pyinstaller cant package >=0.10.0.
See https://github.com/librosa/librosa/issues/1705.
"""
import os
from pathlib import Path
from shutil import copytree

import PyInstaller.__main__

PyInstaller.__main__.run(
    [
        "--icon=gui/img/birdnet-icon.ico",
        "--name=BirdNET-Analyzer-Analysis",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--add-data=eBird_taxonomy_codes_2021E.json" + os.pathsep + ".",
        "--add-data=checkpoints" + os.pathsep + "checkpoints",
        "--add-data=example/soundscape.wav" + os.pathsep + "example",
        "--add-data=example/species_list.txt" + os.pathsep + "example",
        "--add-data=labels" + os.pathsep + "labels",
        "--additional-hooks-dir=extra-hooks",
        "birdnet" + os.path.sep + "analysis" + os.path.sep + "main.py",
    ]
)

if False:
    copytree(
        src=Path('.') / 'dist' / 'BirdNET-Analyzer-Analysis',
        dst=Path('.') / 'build' / 'BirdNET-Analyzer-Analysis',
        dirs_exist_ok=True,
    )
