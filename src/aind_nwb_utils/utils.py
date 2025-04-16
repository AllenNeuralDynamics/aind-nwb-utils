"""Utility functions for working with NWB files."""

import datetime
from pathlib import Path
from typing import Union

import pynwb
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO

from aind_nwb_utils.nwb_io import create_temp_nwb, determine_io

def get_nwb_attribute(
    main_io: Union[NWBHDF5IO, NWBZarrIO], sub_io: Union[NWBHDF5IO, NWBZarrIO]
) -> Union[NWBHDF5IO, NWBZarrIO]:
    """Get an attribute from the NWB file

    Parameters
    ----------
    main_io : Union[NWBHDF5IO, NWBZarrIO]
        the io object
    sub_io : Union[NWBHDF5IO, NWBZarrIO]
        the sub io object

    Returns
    -------
    Any
        the attribute
    """
    for field_name in sub_io.fields.keys():
        attribute = getattr(sub_io, field_name)

        # Skip non-container attributes
        if isinstance(attribute, (str, datetime.datetime, list, pynwb.file.Subject)):
            continue

        # If the attribute is TimeIntervals, handle it separately
        if isinstance(attribute, pynwb.epoch.TimeIntervals):
            attribute.reset_parent()
            attribute.parent = main_io  # 🔹 Explicitly set the parent to avoid orphan errors
            if field_name == "intervals":
                main_io.add_time_intervals(attribute)
            continue  # Skip further processing

        # Handle dictionary-like attributes
        if hasattr(attribute, "items"):
            for name, data in attribute.items():
                data.reset_parent()
                data.parent = main_io  # 🔹 Explicitly set parent here as well
                if name not in getattr(main_io, field_name, {}):
                    if field_name == "acquisition":
                        main_io.add_acquisition(data)
                    elif field_name == "processing":
                        main_io.add_processing_module(data)
                    elif field_name == "analysis":
                        main_io.add_analysis(data)
                    elif field_name == "intervals":
                        main_io.add_time_intervals(data)
                    else:
                        raise ValueError(f"Unknown attribute type: {field_name}")
        else:
            raise TypeError(f"Unexpected type for {field_name}: {type(attribute)}")

    return main_io

def combine_nwb_file(main_nwb_fp: Path, sub_nwb_fp: Path, save_dir: Path, save_io) -> Path:
    """Combine two NWB files and save to scratch directory

    Parameters
    ----------
    main_nwb_fp : Path
        path to the main NWB file
    sub_nwb_fp : Path
        path to the sub NWB file
    save_io : Union[NWBHDF5IO, NWBZarrIO]
        how to save the nwb
    Returns
    -------
    Path
        the path to the saved nwb
    """
    main_io = determine_io(main_nwb_fp)
    sub_io = determine_io(sub_nwb_fp)
    scratch_fp = create_temp_nwb(save_dir, save_io)
    with main_io(main_nwb_fp, "r") as main_io:
        main_nwb = main_io.read()
        with sub_io(sub_nwb_fp, "r") as read_io:
            sub_nwb = read_io.read()
            main_nwb = get_nwb_attribute(main_nwb, sub_nwb)
            with save_io(scratch_fp, "w") as io:
                io.export(src_io=main_io, write_args=dict(link_data=False))
    return scratch_fp
