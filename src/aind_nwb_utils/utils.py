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
    """Merge attributes from sub_io into main_io."""

    def is_non_mergeable(attr):
        return isinstance(attr, (str, datetime.datetime, list, pynwb.file.Subject))

    def add_data(field: str, name: str, obj):
        obj.reset_parent()
        obj.parent = main_io
        existing = getattr(main_io, field, {})
        if name in existing:
            return
        if field == "acquisition":
            main_io.add_acquisition(obj)
        elif field == "processing":
            main_io.add_processing_module(obj)
        elif field == "analysis":
            main_io.add_analysis(obj)
        elif field == "intervals":
            main_io.add_time_intervals(obj)
        else:
            raise ValueError(f"Unknown attribute type: {field}")

    for field_name in sub_io.fields.keys():
        attr = getattr(sub_io, field_name)

        if is_non_mergeable(attr):
            continue

        if isinstance(attr, pynwb.epoch.TimeIntervals):
            attr.reset_parent()
            attr.parent = main_io
            if field_name == "intervals":
                main_io.add_time_intervals(attr)
            continue

        if hasattr(attr, "items"):
            for name, data in attr.items():
                add_data(field_name, name, data)
        else:
            raise TypeError(f"Unexpected type for {field_name}: {type(attr)}")

    return main_io


def combine_nwb_file(
    main_nwb_fp: Path, sub_nwb_fp: Path, save_dir: Path, save_io
) -> Path:
    """Combine two NWB files and save to scratch directory

    Parameters
    ----------
    main_nwb_fp : Path
        path to the main NWB file
    sub_nwb_fp : Path
        path to the sub NWB file
    save_dir : Path
        path to the save location for the NWB file
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
