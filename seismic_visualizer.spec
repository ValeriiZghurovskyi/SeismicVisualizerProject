# -*- mode: python ; coding: utf-8 -*-
import glob, site

block_cipher = None

# Find mypyc compiled extension regardless of Python version or environment
_mypyc_bins = []
for _sp in site.getsitepackages():
    _found = glob.glob(f'{_sp}/*__mypyc*.pyd')
    _mypyc_bins += [(_f, '.') for _f in _found]

a = Analysis(
    ['src/seismic_visualizer/__main__.py'],
    pathex=['.'],
    binaries=_mypyc_bins,
    datas=[
        ('models', 'models'),
        ('src/seismic_visualizer/ui/resources', 'seismic_visualizer/ui/resources'),
    ],
    hiddenimports=[
        '0aca9ce3d91742c5b361__mypyc',
        'onnxruntime',
        'onnxruntime.capi._pybind_state',
        'onnxruntime.backend',
        'scipy._lib.messagestream',
        'scipy.special._cdflib',
        'scipy.special._ufuncs',
        'shapely',
        'shapely.geometry',
        'shapely.ops',
        'pyvista',
        'vtkmodules.all',
        'vtkmodules.util.numpy_support',
        'vtkmodules.qt.QVTKRenderWindowInteractor',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/rthook_onnxruntime.py'],
    excludes=['tkinter', 'matplotlib.tests', 'numpy.tests', 'torch'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    collect_all=['onnxruntime', 'segyio'],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SeismicVisualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SeismicVisualizer',
)
