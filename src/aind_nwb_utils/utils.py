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
from aind_nwb_utils.nwb_io import determine_io


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
    return isinstance(attr, (str,
                             datetime.datetime,
                             list,
                             pynwb.file.Subject,),)


def downcast_timeseries_precision(ts_obj: TimeSeries) -> TimeSeries:
    """
    Cast TimeSeries data from float64/int64 to float32/int32 and return new object.

    Parameters
    ----------
    ts_obj : TimeSeries
        The TimeSeries object to downcast.

    Returns
    -------
    TimeSeries
        A new TimeSeries object with downcasted data, or the original if no casting needed.
    """
    data = ts_obj.data
    if hasattr(data, "dtype") and data.dtype in [np.float64, np.int64]:
        try:
            new_dtype = np.float32
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


def cast_vectordata_if_needed(obj):
    """
    Cast the data inside VectorData objects if necessary.

    Parameters
    ----------
    obj : Any
        The object to check and potentially cast.
    Returns
    -------
    Any
        The original object or a new VectorData with casted data.
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
    main_io: Union[NWBHDF5IO, NWBZarrIO],
    field: str,
    name: str,
    obj: Any
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

        if isinstance(attr, dict) or hasattr(attr, "keys"):
            for name, data in attr.items():
                if isinstance(data, TimeSeries):
                    data = downcast_timeseries_precision(data)
                data = cast_vectordata_if_needed(data)

                if field_name == "devices":
                    if name not in main_io.devices:
                        data.reset_parent()
                        data.parent = main_io
                        main_io.add_device(data)
                    continue

                add_data(main_io, field_name, name, data)
        else:
            raise TypeError(f"Unexpected type for {field_name}: {type(attr)}")

    return main_io


def combine_nwb_file(
    main_nwb_fp: Path,
    sub_nwb_fp: Path,
    output_path: Path,
    save_io: Union[NWBHDF5IO, NWBZarrIO],
) -> Path:
    """
    Combine two NWB files by merging attributes from a
    secondary file into a main file, and write to output_path.

    Parameters
    ----------
    main_nwb_fp : Path
        Path to the main NWB file.
    sub_nwb_fp : Path
        Path to the secondary NWB file whose data will be merged.
    output_path : Path
        Path to write the merged NWB file.
    save_io : Union[NWBHDF5IO, NWBZarrIO]
        IO class used to write the resulting NWB file.

    Returns
    -------
    Path
        Path to the saved combined NWB file.
    """
    main_io_class = determine_io(main_nwb_fp)
    sub_io_class = determine_io(sub_nwb_fp)

    print(main_nwb_fp)
    print(sub_nwb_fp)
    print(f"Saving merged file to: {output_path}")

    with main_io_class(main_nwb_fp, "r") as main_io:
        main_nwb = main_io.read()

        with sub_io_class(sub_nwb_fp, "r") as sub_io:
            sub_nwb = sub_io.read()
            main_nwb = get_nwb_attribute(main_nwb, sub_nwb)

            with save_io(output_path, "w") as out_io:
                out_io.export(src_io=main_io, write_args=dict(link_data=False))

    return output_path
