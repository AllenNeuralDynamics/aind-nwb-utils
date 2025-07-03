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


# Global flag to control aggressive dtype casting
ENABLE_DTYPE_CASTING = False


def set_dtype_casting(enabled: bool):
    """
    Enable or disable aggressive dtype casting.

    Parameters
    ----------
    enabled : bool
        Whether to enable dtype casting from 64-bit to 32-bit types.
    """
    global ENABLE_DTYPE_CASTING
    ENABLE_DTYPE_CASTING = enabled


def is_safe_to_cast(data_array, target_dtype):
    """
    Check if it's safe to cast data to target dtype without losing information.

    Parameters
    ----------
    data_array : np.ndarray
        The data to check
    target_dtype : np.dtype
        The target dtype to cast to

    Returns
    -------
    bool
        True if casting is safe, False otherwise
    """
    try:
        if target_dtype == np.float32:
            # Check if all values are finite and within float32 range
            return (
                np.all(np.isfinite(data_array))
                and np.all(np.abs(data_array) <= np.finfo(np.float32).max)
            )
        elif target_dtype == np.int32:
            # Check if all values are within int32 range
            return (
                np.all(data_array >= np.iinfo(np.int32).min)
                and np.all(data_array <= np.iinfo(np.int32).max)
            )
        return False
    except (ValueError, OverflowError, MemoryError):
        return False


def handle_zarr_dtype_compatibility(data_array, target_dtype):
    """
    Handle Zarr-specific dtype compatibility issues.

    Parameters
    ----------
    data_array : np.ndarray
        The data array to check
    target_dtype : np.dtype
        The target dtype

    Returns
    -------
    bool
        True if the conversion is safe for Zarr, False otherwise
    """
    try:
        # Check if the byte size is compatible
        source_itemsize = data_array.dtype.itemsize
        target_itemsize = np.dtype(target_dtype).itemsize

        # Zarr requires that when changing to smaller dtype,
        # the size must be a divisor of the original size
        if target_itemsize < source_itemsize and source_itemsize % target_itemsize != 0:
            return False

        # Additional safety check for data range
        return is_safe_to_cast(data_array, target_dtype)
    except Exception:
        return False


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
    if not isinstance(ts_obj, TimeSeries) or not ENABLE_DTYPE_CASTING:
        return ts_obj

    data = ts_obj.data
    if hasattr(data, "dtype") and data.dtype in [np.float64, np.int64]:
        try:
            # Check if data size is compatible with smaller dtype
            data_array = np.asarray(data)
            target_dtype = np.float32 if data.dtype == np.float64 else np.int32
            
            # Use Zarr-compatible check
            if handle_zarr_dtype_compatibility(data_array, target_dtype):
                casted_data = data_array.astype(target_dtype)
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
    if not ENABLE_DTYPE_CASTING or not isinstance(obj, VectorData) or not hasattr(obj, "data"):
        return obj
        
    dtype = getattr(obj.data, "dtype", None)
    if dtype in [np.float64, np.int64]:
        try:
            data_array = np.asarray(obj.data)
            target_dtype = np.float32 if dtype == np.float64 else np.int32
            
            # Use Zarr-compatible check
            if handle_zarr_dtype_compatibility(data_array, target_dtype):
                obj.data = data_array.astype(target_dtype)
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
        
    Notes
    -----
    If you encounter dtype casting errors with Zarr files, you can disable
    automatic dtype casting by calling:
    
        from aind_nwb_utils import set_dtype_casting
        set_dtype_casting(False)
        
    before calling this function.
    """
    main_io = determine_io(main_nwb_fp)
    sub_io = determine_io(sub_nwb_fp)
    scratch_fp = create_temp_nwb(save_dir, save_io)
    
    try:
        with main_io(main_nwb_fp, "r") as main_io:
            main_nwb = main_io.read()
            with sub_io(sub_nwb_fp, "r") as read_io:
                sub_nwb = read_io.read()
                main_nwb = get_nwb_attribute(main_nwb, sub_nwb)
                with save_io(scratch_fp, "w") as io:
                    io.export(src_io=main_io, write_args=dict(link_data=False))
    except ValueError as e:
        if "divisor" in str(e) and "dtype" in str(e):
            print("Error: Dtype casting issue detected. "
                  "Consider disabling dtype casting with set_dtype_casting(False)")
        raise
    
    return scratch_fp
