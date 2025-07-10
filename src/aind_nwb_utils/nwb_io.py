"""io tools for nwb files"""

import atexit
import os
import tempfile
from pathlib import Path
from typing import Union

from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO


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


def determine_io(nwb_path: Path) -> Union[NWBHDF5IO, NWBZarrIO]:
    """determine the io type

    Returns
    -------
    Union[NWBHDF5IO, NWBZarrIO]
        the appropriate io object
    """
    if nwb_path.is_dir():
        return NWBZarrIO
    return NWBHDF5IO
