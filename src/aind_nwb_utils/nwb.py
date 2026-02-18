from contextlib import ExitStack
from typing import Union
from pathlib import Path

import pynwb
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
import logging

from aind_nwb_utils.nwb_io import determine_io
from aind_nwb_utils.utils import merge_nwb_attribute

logger = logging.getLogger(__name__)


class NWBCombineIO:
    """
    Merges sub NWB files into a main NWB file.

    Supports two usage patterns:

    As a context manager::

        with NWBCombineIO(main_path, [sub_path]) as (nwb_file, main_io):
            # merged NWBFile available with all IO handles open
            ...

    As a standalone object::

        combiner = NWBCombineIO(main_path, [sub_path])
        nwb = combiner.read()
        # ... work with nwb ...
        combiner.close()

    Parameters
    ----------
    main_nwb_fp : Path
        Path to the main NWB file.
    sub_nwb_paths : list[Path]
        List of paths to the secondary NWB files
        whose data will be merged.
    """

    _IO_FORMATS = {
        "zarr": NWBZarrIO,
        "hdf5": NWBHDF5IO,
    }

    def __init__(
        self,
        main_nwb_fp: Path,
        sub_nwb_paths: list[Path],
    ) -> None:
        self._main_nwb_fp = main_nwb_fp
        self._sub_nwb_paths = sub_nwb_paths
        self._stack: ExitStack | None = None
        self._main_io: Union[NWBHDF5IO, NWBZarrIO] | None = None
        self._nwb: pynwb.NWBFile | None = None

    def _open(self) -> tuple[pynwb.NWBFile, Union[NWBHDF5IO, NWBZarrIO]]:
        """Open all IO handles and perform the merge."""
        if self._nwb is not None:
            return self._nwb, self._main_io

        self._stack = ExitStack()
        main_io_class = determine_io(self._main_nwb_fp)

        logger.info(self._main_nwb_fp)
        self._main_io = self._stack.enter_context(
            main_io_class(self._main_nwb_fp, "r")
        )
        self._nwb = self._main_io.read()

        for sub_nwb_fp in self._sub_nwb_paths:
            logger.info(sub_nwb_fp)
            sub_io_class = determine_io(sub_nwb_fp)
            sub_io = self._stack.enter_context(sub_io_class(sub_nwb_fp, "r"))
            sub_nwb = sub_io.read()
            self._nwb = merge_nwb_attribute(self._nwb, sub_nwb)

        return self._nwb, self._main_io

    def read(self) -> pynwb.NWBFile:
        """
        Merge and return the combined NWBFile.

        IO handles remain open so lazy-loaded data is accessible.
        Call :meth:`close` when finished.

        Returns
        -------
        pynwb.NWBFile
            The combined NWB file.
        """
        nwb, _ = self._open()
        return nwb

    def write(self, output_path: Path, format: str = "zarr") -> None:
        """
        Export the combined NWB file to disk.

        Parameters
        ----------
        output_path : Path
            Path to write the exported NWB file.
        format : str
            Output format. Either ``"zarr"`` or ``"hdf5"``.
            Defaults to ``"zarr"``.
        """
        if self._main_io is None:
            self._open()

        save_io = self._IO_FORMATS.get(format.lower())
        if save_io is None:
            raise ValueError(
                f"Unknown format '{format}'. "
                f"Supported formats: {list(self._IO_FORMATS.keys())}"
            )

        logger.info(f"Writing to disk at {output_path}")
        with save_io(output_path, "w") as out_io:
            out_io.export(
                src_io=self._main_io, write_args=dict(link_data=False)
            )

    def close(self) -> None:
        """Close all open IO handles."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def __enter__(
        self,
    ) -> tuple[pynwb.NWBFile, Union[NWBHDF5IO, NWBZarrIO]]:
        """Open IO handles, merge files, and return the result.

        Returns
        -------
        tuple[pynwb.NWBFile, Union[NWBHDF5IO, NWBZarrIO]]
            The combined NWB file and the main IO handle.
        """
        return self._open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close all open IO handles on context manager exit."""
        self.close()
