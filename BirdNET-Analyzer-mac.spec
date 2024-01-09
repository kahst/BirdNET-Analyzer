# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('eBird_taxonomy_codes_2021E.json', '.'), ('checkpoints', 'checkpoints'), ('example/soundscape.wav', 'example'), ('example/species_list.txt', 'example'), ('labels', 'labels')],
    hiddenimports=[],
    hookspath=['extra-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
        'tensorflow': 'py'
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BirdNET-Analyzer-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gui/img/birdnet-icon.ico'],
)
app = BUNDLE(
    exe,
    name='BirdNET-Analyzer-GUI.app',
    icon='gui/img/birdnet-icon.ico',
    bundle_identifier=None,
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
    },
)
