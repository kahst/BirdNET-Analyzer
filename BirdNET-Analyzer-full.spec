# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


analyzer = Analysis(
    ['birdnet/analysis/main.py'],
    pathex=[],
    binaries=[],
    datas=[('eBird_taxonomy_codes_2021E.json', '.'), ('checkpoints', 'checkpoints'), ('example/soundscape.wav', 'example'), ('example/species_list.txt', 'example'), ('labels', 'labels')],
    hiddenimports=[],
    hookspath=['extra-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
analyzer_pyz = PYZ(analyzer.pure, analyzer.zipped_data, cipher=block_cipher)

analyzer_exe = EXE(
    analyzer_pyz,
    analyzer.scripts,
    [],
    exclude_binaries=True,
    name='BirdNET-Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gui\\img\\birdnet-icon.ico'],
)

gui = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('eBird_taxonomy_codes_2021E.json', '.'), ('checkpoints', 'checkpoints'), ('example/soundscape.wav', 'example'), ('example/species_list.txt', 'example'), ('labels', 'labels')],
    hiddenimports=[],
    hookspath=['extra-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
gui_pyz = PYZ(gui.pure, gui.zipped_data, cipher=block_cipher)

gui_exe = EXE(
    gui_pyz,
    gui.scripts,
    [],
    exclude_binaries=True,
    name='BirdNET-Analyzer-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gui\\img\\birdnet-icon.ico'],
)


coll = COLLECT(
    analyzer_exe,
    analyzer.binaries,
    analyzer.zipfiles,
    analyzer.datas,
    gui_exe,
    gui.binaries,
    gui.zipfiles,
    gui.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BirdNET-Analyzer',
)
