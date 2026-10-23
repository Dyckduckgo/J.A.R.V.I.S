# PyInstaller spec for a one-folder Windows build of J.A.R.V.I.S.
# Build with:  pyinstaller --noconfirm --clean packaging/jarvis.spec
# Output:      dist/JARVIS/JARVIS.exe
import os

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 - SPECPATH is injected by PyInstaller

hiddenimports = collect_submodules("jarvis") + collect_submodules("keyring.backends")
datas = copy_metadata("keyring") + copy_metadata("groq")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],  # noqa: F821
    pathex=[os.path.join(ROOT, "src")],
    hiddenimports=hiddenimports,
    datas=datas,
    excludes=[
        "tkinter",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.Qt3DCore",
        "PySide6.QtMultimedia",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="JARVIS",
    console=False,
    disable_windowed_traceback=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="JARVIS")
