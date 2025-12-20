# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['anonim_gui_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['spacy.lang.es', 'spacy.tokens.span_group', 'anonim_meddocan_real_1'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ANONIM_LaPaz',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
