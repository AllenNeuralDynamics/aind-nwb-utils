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


def safe_dtype_cast(data, target_dtype):
    """
    Safely cast data to target dtype, handling various array layouts.
    
    Parameters
    ----------
    data : array-like
        The data to cast
    target_dtype : numpy.dtype
        The target data type
        
    Returns
    -------
    numpy.ndarray
        The safely cast data
    """
    try:
        # Convert to numpy array first if not already
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Create a copy with the new dtype to avoid memory layout issues
        return np.array(data, dtype=target_dtype, copy=True)
    except (ValueError, TypeError) as e:
        print(f"Could not cast data to {target_dtype}: {e}")
        return data


def cast_timeseries_if_needed(ts_obj):
    """
    Cast TimeSeries data to smaller precision if possible.
    
    Parameters
    ----------
    ts_obj : TimeSeries or other
        The object to potentially cast
        
    Returns
    -------
    TimeSeries or original object
        The cast TimeSeries or original object if casting not needed/possible
    """
    if not isinstance(ts_obj, TimeSeries):
        return ts_obj

    data = ts_obj.data
    if hasattr(data, "dtype") and data.dtype in [np.float64, np.int64]:
        try:
            new_dtype = np.float32 if data.dtype == np.float64 else np.int32
            casted_data = safe_dtype_cast(data, new_dtype)
            
            # Only create new TimeSeries if casting was successful
            if casted_data.dtype == new_dtype:
                return TimeSeries(
                    name=ts_obj.name,
                    data=casted_data,
                    unit=ts_obj.unit,
                    timestamps=ts_obj.timestamps,
                    description=ts_obj.description,
                )
        except Exception as e:
            print(f"Could not cast TimeSeries '{ts_obj.name}': {e}")
    
    return ts_obj


def cast_vectordata_if_needed(obj):
    """
    Cast the data inside VectorData objects if necessary.
    
    Parameters
    ----------
    obj : VectorData or other
        The object to potentially cast
        
    Returns
    -------
    VectorData or original object
        The object with potentially cast data
    """
    if isinstance(obj, VectorData) and hasattr(obj, "data"):
        dtype = getattr(obj.data, "dtype", None)
        if dtype in [np.float64, np.int64]:
            try:
                new_dtype = np.float32 if dtype == np.float64 else np.int32
                casted_data = safe_dtype_cast(obj.data, new_dtype)
                
                # Only update if casting was successful
                if casted_data.dtype == new_dtype:
                    obj.data = casted_data
            except Exception as e:
                print(f"Could not cast VectorData '{obj.name}': {e}")
    
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
    main_io = determine_io(main_nwb_fp)
    sub_io = determine_io(sub_nwb_fp)
    scratch_fp = create_temp_nwb(save_dir, save_io)
    
    with main_io(main_nwb_fp, "r") as main_file_io:
        main_nwb = main_file_io.read()
        with sub_io(sub_nwb_fp, "r") as read_io:
            sub_nwb = read_io.read()
            main_nwb = get_nwb_attribute(main_nwb, sub_nwb)
            with save_io(scratch_fp, "w") as write_io:
                write_io.export(src_io=main_file_io, write_args=dict(link_data=False))
    
    return scratch_fp
