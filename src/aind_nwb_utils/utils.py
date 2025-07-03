"""Utility functions for working with NWB files."""

import datetime
from pathlib import Path
from typing import Union, Any

import numpy as np
import pynwb
from pynwb import TimeSeries
from pynwb.base import VectorData
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO

from aind_nwb_utils.nwb_io import create_temp_nwb, determine_io


def is_non_mergeable(attr: Any):
    """
    Check if an attribute is not suitable for merging into the NWB file.

    Parameters
    ----------
    attr : Any
        The attribute to check.

    Returns
    -------
    bool
        True if the attribute is a non-container type or
        should be skipped during merging.
    """
    return isinstance(
        attr,
        (
            str,
            datetime.datetime,
            list,
            pynwb.file.Subject,
        ),
    )


def cast_timeseries_if_needed(ts_obj):
    if not isinstance(ts_obj, TimeSeries):
        return ts_obj

    data = ts_obj.data
    if hasattr(data, "dtype") and data.dtype in [np.float64, np.int64]:
        try:
            new_dtype = np.float32 if data.dtype == np.float64 else np.int32
            casted_data = np.asarray(data).astype(new_dtype)

            return TimeSeries(
                name=ts_obj.name,
                data=casted_data,
                unit=ts_obj.unit,
                timestamps=ts_obj.timestamps,
                description=ts_obj.description,
            )
        except Exception as e:
            print(f"Could not cast TimeSeries '{ts_obj.name}' — {e}")
    return ts_obj

def cast_vectordata_if_needed(obj):
    """
    Cast the data inside VectorData objects if necessary.
    """
    if isinstance(obj, VectorData) and hasattr(obj, "data"):
        dtype = getattr(obj.data, "dtype", None)
        if dtype in [np.float64, np.int64]:
            try:
                new_dtype = np.float32 if dtype == np.float64 else np.int32
                obj.data = np.asarray(obj.data).astype(new_dtype)
            except Exception as e:
                print(f"Could not cast VectorData '{obj.name}' — {e}")
    return obj


def add_data(
    main_io: Union[NWBHDF5IO, NWBZarrIO], field: str, name: str, obj: Any
):
    """
    Add a data object to the appropriate field in the NWB file.

    Parameters
    ----------
    main_io : Union[NWBHDF5IO, NWBZarrIO]
        The main NWB file IO object to add data to.
    field : str
        The field of the NWB file to add to
        (e.g., 'acquisition', 'processing').
    name : str
        The name of the object to be added.
    obj : Any
        The NWB container object to add.
    """
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


def get_nwb_attribute(
    main_io: Union[NWBHDF5IO, NWBZarrIO], sub_io: Union[NWBHDF5IO, NWBZarrIO]
) -> Union[NWBHDF5IO, NWBZarrIO]:
    """
    Merge container-type attributes from one NWB file
        (sub_io) into another (main_io).

    Parameters
    ----------
    main_io : Union[NWBHDF5IO, NWBZarrIO]
        The destination NWB file IO object.
    sub_io : Union[NWBHDF5IO, NWBZarrIO]
        The source NWB file IO object to merge from.

    Returns
    -------
    Union[NWBHDF5IO, NWBZarrIO]
        The modified main_io with attributes from sub_io merged in.
    """
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
                data = cast_timeseries_if_needed(data)
                data = cast_vectordata_if_needed(data)
                add_data(main_io, field_name, name, data)
        else:
            raise TypeError(f"Unexpected type for {field_name}: {type(attr)}")

    return main_io


def combine_nwb_file(main_nwb_fp, sub_nwb_fp, save_dir, save_io):
    main_io = determine_io(main_nwb_fp)
    sub_io = determine_io(sub_nwb_fp)
    scratch_fp = create_temp_nwb(save_dir, save_io)
    temp_fp = create_temp_nwb(save_dir, save_io)  # temp file for merged NWBFile

    with main_io(main_nwb_fp, "r") as main_io_obj, sub_io(sub_nwb_fp, "r") as sub_io_obj:
        main_nwb = main_io_obj.read()
        sub_nwb = sub_io_obj.read()
        main_nwb = get_nwb_attribute(main_nwb, sub_nwb)

    # Write merged NWBFile to temp file
    with save_io(temp_fp, "w") as temp_io:
        temp_io.write(main_nwb)

    # Now open temp file and export to scratch_fp
    with save_io(temp_fp, "r") as temp_io, save_io(scratch_fp, "w") as out_io:
        out_io.export(src_io=temp_io, write_args=dict(link_data=False))

    return scratch_fp
