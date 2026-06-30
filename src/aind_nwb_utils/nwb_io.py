"""io tools for nwb files"""

import atexit
import os
import tempfile
from pathlib import Path
from typing import Union

from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO

# An HDF5 file always begins with this 8-byte signature.
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
# Keys that mark the root of a Zarr store (v2, consolidated, and v3).
_ZARR_ROOT_KEYS = (".zgroup", ".zmetadata", "zarr.json")


def create_temp_nwb(
    save_dir: str, save_strategy: Union[NWBHDF5IO, NWBZarrIO]
) -> Path:
    """Create a temporary file and return the path

    Parameters
    ----------
    save_strategy : Union[NWBHDF5IO, NWBZarrIO]
        to determine if a temp file or directory should be created
    save_dir : str
        option to specify root directory
    Returns
    -------
    str
        the path to the temporary file
    """
    if save_strategy is NWBZarrIO:
        temp_path = tempfile.mkdtemp(dir=save_dir)
    else:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".nwb")
        temp_path = temp.name
        temp.close()

    atexit.register(os.unlink, temp_path)
    return Path(temp_path)


def _detect_by_structure(
    nwb_path: Path, path_str: str, is_s3: bool
) -> Union[NWBHDF5IO, NWBZarrIO, None]:
    """Detect the io type by inspecting the file/store contents.

    This is the most reliable signal because it does not depend on naming
    conventions. Returns ``None`` when the structure cannot be inspected
    (e.g. the path does not exist, is empty, or S3 access is unavailable)
    so the caller can fall back to other heuristics.

    Parameters
    ----------
    nwb_path : Path
        the path to inspect
    path_str : str
        the path coerced to a string
    is_s3 : bool
        whether ``path_str`` is an ``s3://`` URI

    Returns
    -------
    Union[NWBHDF5IO, NWBZarrIO, None]
        the detected io object, or ``None`` if undetermined
    """
    if is_s3:
        return _detect_s3_by_structure(path_str)

    if os.path.isdir(nwb_path):
        root = Path(nwb_path)
        if any((root / key).exists() for key in _ZARR_ROOT_KEYS):
            return NWBZarrIO
        return None

    try:
        with open(nwb_path, "rb") as handle:
            if handle.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE:
                return NWBHDF5IO
    except OSError:
        return None
    return None


def _detect_s3_by_structure(
    path_str: str,
) -> Union[NWBHDF5IO, NWBZarrIO, None]:
    """Detect the io type of an S3 path by inspecting its contents.

    Best-effort: requires ``fsspec``/``s3fs``. Returns ``None`` if those
    are unavailable or the structure is inconclusive, so the caller can
    fall back to the filename extension.

    Parameters
    ----------
    path_str : str
        the ``s3://`` URI to inspect

    Returns
    -------
    Union[NWBHDF5IO, NWBZarrIO, None]
        the detected io object, or ``None`` if undetermined
    """
    try:
        import fsspec
    except ImportError:
        return None

    try:
        fs, _ = fsspec.core.url_to_fs(path_str)
        root = path_str.rstrip("/")
        if any(fs.exists(f"{root}/{key}") for key in _ZARR_ROOT_KEYS):
            return NWBZarrIO
        with fs.open(path_str, "rb") as handle:
            if handle.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE:
                return NWBHDF5IO
    except Exception:
        # Best-effort probe: any failure (missing object, network, or
        # credential/auth errors) falls back to extension-based detection.
        # A genuine access problem will surface at actual read time.
        return None
    return None


def determine_io(nwb_path: Path) -> Union[NWBHDF5IO, NWBZarrIO]:
    """determine the io type

    Detection is layered, strongest signal first:

    1. The file/store structure (HDF5 signature or Zarr root keys). This
       is robust regardless of how the path is named.
    2. The filename extension (``.nwb`` -> HDF5, ``.zarr`` -> Zarr).
    3. A final heuristic: directories are Zarr stores, everything else
       is treated as HDF5.

    Parameters
    ----------
    nwb_path : Path
        the path to the nwb file or store

    Returns
    -------
    Union[NWBHDF5IO, NWBZarrIO]
        the appropriate io object
    """
    path_str = str(nwb_path)
    is_s3 = path_str.startswith("s3://")

    io_class = _detect_by_structure(nwb_path, path_str, is_s3)
    if io_class is not None:
        return io_class

    if path_str.endswith(".nwb"):
        return NWBHDF5IO
    if path_str.endswith(".zarr"):
        return NWBZarrIO

    if not is_s3 and os.path.isdir(nwb_path):
        return NWBZarrIO
    return NWBHDF5IO
