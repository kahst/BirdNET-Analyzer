"""Packages the analyzer and the gui into an application.
Librosa 0.9.2 has to be used, since pyinstaller cant package >=0.10.0.
See https://github.com/librosa/librosa/issues/1705.
"""
import PyInstaller.__main__


PyInstaller.__main__.run(["--clean", "--noconfirm", "BirdNET-Analyzer-full.spec"])
