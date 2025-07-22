"""Utility functions for working with NWB files."""

import datetime
import json
import uuid
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Union

import pynwb
import pytz
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from pynwb.file import Subject

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
    with main_io(main_nwb_fp, "r") as main_io:
        main_nwb = main_io.read()
        with sub_io(sub_nwb_fp, "r") as read_io:
            sub_nwb = read_io.read()
            main_nwb = get_nwb_attribute(main_nwb, sub_nwb)
            with save_io(scratch_fp, "w") as io:
                io.export(src_io=main_io, write_args=dict(link_data=False))
    return scratch_fp


def _get_session_start_date_time(session_start_date_string: str) -> datetime:
    """
    Returns the datetime given the string

    Parameters
    ----------
    session_start_date_string: str
        The session start date as a string

    Returns
    -------
    datetime
        The session start datetime object
    """
    # ported this from subject nwb capsule
    date_format_no_tz = "%Y-%m-%dT%H:%M:%S"
    date_format_tz = "%Y-%m-%dT%H:%M:%S%z"
    date_format_frac_tz = "%Y-%m-%dT%H:%M:%S.%f%z"
    supported_date_formats = [
        date_format_no_tz,
        date_format_tz,
        date_format_frac_tz,
    ]

    # Use strptime to parse the string into a datetime object
    # not sure if this needs to go through all supported formats?
    session_start_date_time = None
    for date_format in supported_date_formats:
        try:
            session_start_date_time = dt.strptime(
                session_start_date_string, date_format
            )
            break
        except Exception:
            pass

    if session_start_date_time.tzinfo is None:
        pacific = pytz.timezone("US/Pacific")
        session_start_date_time = pacific.localize(session_start_date_time)

    return session_start_date_time


def get_subject_nwb_object(
    data_description: dict[str, Any], subject_metadata: dict[str, Any]
) -> Subject:
    """
    Return the NWB Subject object made from the metadata files

    Parameters
    ----------
    data_description : dict[str, Any]
        Data description json file

    subject_metadata: dict[str, Any]
        Subject metadata json file

    Returns
    -------
    Subject
        The Subject object containing metadata such as subject ID,
        species, sex, date of birth, and other experimental details.
    """

    session_start_date_string = data_description["creation_time"]
    dob = subject_metadata["date_of_birth"]
    subject_dob = dt.strptime(dob, "%Y-%m-%d").replace(
        tzinfo=pytz.timezone("US/Pacific")
    )

    session_start_date_time = _get_session_start_date_time(
        session_start_date_string
    )

    subject_age = session_start_date_time - subject_dob

    age = "P" + str(subject_age.days) + "D"
    if isinstance(subject_metadata["species"], dict):
        species = subject_metadata["species"]["name"]
    else:
        species = subject_metadata["species"]

    return Subject(
        subject_id=subject_metadata["subject_id"],
        species=species,
        sex=subject_metadata["sex"][0].upper(),
        date_of_birth=subject_dob,
        age=age,
        genotype=subject_metadata["genotype"],
        description=None,
        strain=subject_metadata.get("background_strain")
        or subject_metadata.get("breeding_group"),
    )


def create_base_nwb_file(data_path: Path) -> pynwb.NWBFile:
    """
    Creates the base nwb file given the path to the metadata files

    Parameters
    ----------
    data_path: Path
        The path with the relevant metadata files

    Returns
    -------
    pynwb.NWBFile
        The base nwb file with subject metadata
    """
    data_description_path = data_path / "data_description.json"
    subject_json_path = data_path / "subject.json"

    if not data_description_path.exists():
        raise FileNotFoundError(
            f"No data description json found at {data_description_path}"
        )

    if not subject_json_path.exists():
        raise FileNotFoundError(
            f"No subject json found at {subject_json_path}"
        )

    with open(data_description_path, "r") as f:
        data_description = json.load(f)

    with open(subject_json_path, "r") as f:
        subject_metadata = json.load(f)

    nwb_subject = get_subject_nwb_object(data_description, subject_metadata)
    session_start_date_time = _get_session_start_date_time(
        data_description["creation_time"]
    )

    nwb_file = pynwb.NWBFile(
        session_description="Base NWB file generated with subject metadata",
        identifier=str(uuid.uuid4()),
        session_start_time=session_start_date_time,
        institution=data_description["institution"].get("name", None),
        subject=nwb_subject,
        session_id=data_description["name"],
    )

    return nwb_file
