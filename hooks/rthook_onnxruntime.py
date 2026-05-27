import ctypes
import os
import pathlib
import sys

if hasattr(sys, '_MEIPASS'):
    meipass = pathlib.Path(sys._MEIPASS)
    capi_dir = meipass / 'onnxruntime' / 'capi'

    # Register DLL search directories
    os.add_dll_directory(str(meipass))
    if capi_dir.exists():
        os.add_dll_directory(str(capi_dir))

    # Prepend to PATH so LoadLibrary also finds them
    os.environ['PATH'] = (
        str(capi_dir) + os.pathsep
        + str(meipass) + os.pathsep
        + os.environ.get('PATH', '')
    )

    # Explicitly preload onnxruntime DLLs before pybind11 state is imported
    for dll_name in ('onnxruntime_providers_shared.dll', 'onnxruntime.dll'):
        dll_path = capi_dir / dll_name
        if dll_path.exists():
            ctypes.CDLL(str(dll_path))
