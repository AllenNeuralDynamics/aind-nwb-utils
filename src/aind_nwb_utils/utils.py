"""Utility functions for working with NWB files."""

import datetime
from pathlib import Path
from typing import Union, Any

import pynwb
import numpy as np
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries
from hdmf.common.table import VectorData
from hdmf.common import DynamicTable

from aind_nwb_utils.nwb_io import create_temp_nwb, determine_io


def cast_timeseries_if_needed(ts_obj):
    """
    If TimeSeries data is float64/int64, cast to float32/int32 and return new object.
    """
    if not isinstance(ts_obj, TimeSeries):
        return ts_obj  # Only handle TimeSeries

    data = ts_obj.data
    if hasattr(data, "dtype") and data.dtype in [np.float64, np.int64]:
        try:
            new_dtype = np.float32 if data.dtype == np.float64 else np.int32
            casted_data = np.asarray(data).astype(new_dtype)

            return TimeSeries(
                name=ts_obj.name,
                data=casted_data,
                unit=ts_obj.unit,
                rate=ts_obj.rate,
                conversion=ts_obj.conversion,
                resolution=ts_obj.resolution,
                starting_time=ts_obj.starting_time,
                timestamps=ts_obj.timestamps,
                description=ts_obj.description,
                comments=ts_obj.comments,
                control=ts_obj.control,
                control_description=ts_obj.control_description,
            )
        except Exception as e:
            print(f"Could not cast TimeSeries '{ts_obj.name}' — {e}")
    return ts_obj


def cast_vector_data_if_needed(obj):
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


def get_nwb_attribute(
    main_io: Union[NWBHDF5IO, NWBZarrIO], sub_io: Union[NWBHDF5IO, NWBZarrIO]
) -> Union[NWBHDF5IO, NWBZarrIO]:
    """
    Merge container-type attributes from one NWB file (sub_io)
    into another (main_io), with dtype-safe handling.
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
                data = cast_vector_data_if_needed(data)
                add_data(main_io, field_name, name, data)
        else:
            raise TypeError(f"Unexpected type for {field_name}: {type(attr)}")

    return main_io


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

def combine_nwb_file(
    main_nwb_fp: Path,
    sub_nwb_fp: Path,
    save_dir: Path,
    save_io: Union[NWBHDF5IO, NWBZarrIO],
) -> Path:
    """
    Combine two NWB files by merging attributes from a
    secondary file into a main file.

    Parameters
    ----------
    main_nwb_fp : Path
        Path to the main NWB file.
    sub_nwb_fp : Path
        Path to the secondary NWB file whose data will be merged.
    save_dir : Path
        Directory to save the combined NWB file.
    save_io : Union[NWBHDF5IO, NWBZarrIO]
        IO class used to write the resulting NWB file.

    Returns
    -------
    Path
        Path to the saved combined NWB file.
    """
    main_io_cls = determine_io(main_nwb_fp)
    sub_io_cls = determine_io(sub_nwb_fp)
    scratch_fp = create_temp_nwb(save_dir, save_io)

    # Open main and sub NWB IOs for reading
    with main_io_cls(main_nwb_fp, "r") as main_io, sub_io_cls(sub_nwb_fp, "r") as sub_io:
        main_nwb = main_io.read()
        sub_nwb = sub_io.read()

    # Merge sub_nwb into main_nwb
    merged_nwb = get_nwb_attribute(main_nwb, sub_nwb)


    # Reset container_source to allow writing
    merged_nwb.container_source = None

    # Now write the merged NWB to a temporary file using the *save_io* in "w" mode
    with save_io(scratch_fp, "w") as out_io:
        # Write the merged NWBFile directly to disk
        out_io.write(merged_nwb)

    # Finally, export from the saved merged NWB file for your desired export format
    # This means opening the merged file with the *save_io* class and exporting it again
    with save_io(scratch_fp, "r") as final_io, save_io(scratch_fp, "w") as export_io:
        export_io.export(src_io=final_io, write_args=dict(link_data=False))

    return scratch_fp
