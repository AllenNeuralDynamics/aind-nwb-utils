"""Utility functions for working with NWB files."""

import datetime
import json
import uuid
import warnings
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Union

import pynwb
import pytz
from hdmf_zarr import NWBZarrIO
from packaging.version import parse
from pynwb import NWBHDF5IO
from pynwb.file import Device, Subject

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


def get_ephys_devices_from_rig_metadata(  # noqa: C901
    session_folder: str
) -> Union[tuple[dict, dict], tuple[None, None]]:
    """
    Return NWB devices from metadata target locations.

    The schemas used to pupulate the NWBFile and metadata dictionaries are:
    - session.json
    - rig.json

    Parameters
    ----------
    session_folder : str or Path
        The path to the session folder
    segment_index : int
        The segment index to instantiate NWBFile for.
        This is needed to correctly instantiate devices and their locations.

    Returns
    -------
    added_devices: dict (device_name: pynwb.Device) or None
        The instantiated Devices with AIND metadata
    devices_target_location: dict
        Dict with device name to target location
    """
    session_folder = Path(session_folder)
    session_file = session_folder / "acquisition.json"
    rig_file = session_folder / "instrument.json"

    # load json files
    session = None
    if session_file.is_file():
        with open(session_file, "r") as f:
            session = json.load(f)

    rig = None
    if rig_file.is_file():
        with open(rig_file, "r") as f:
            rig = json.load(f)

    # load json files
    session = None
    if session_file.is_file():
        with open(session_file, "r") as f:
            session = json.load(f)

    rig = None
    if rig_file.is_file():
        with open(rig_file, "r") as f:
            rig = json.load(f)

    data_streams = None
    if session is not None:
        session_schema_version = session.get("schema_version", None)

        if session_schema_version is None:
            warnings.warn("Acquisition file does not have schema_version")
            return None, None
        if parse(session_schema_version) >= parse("2.0.0"):
            data_streams = session.get("data_streams", None)
            if data_streams is None:
                warnings.warn("Acquisition file does not have data_streams")
                return None, None
        else:
            warnings.warn(
                f"v{session_schema_version} for acquisition "
                "schema is not currently supported"
            )
            return None, None
    else:
        warnings.warn(f"Acquisition file not found in {session_folder}")
        return None, None

    stimulus_epochs = session.get("stimulus_epochs", None)
    stimulus_device_names = []
    if stimulus_epochs is not None:
        for epoch in stimulus_epochs:
            stimulus_device_names += epoch.get("active_devices", [])

    if rig is not None:
        rig_schema_version = rig.get("schema_version", None)
        if rig_schema_version is None:
            warnings.warn("Instrument file does not have schema_version")
        elif parse(rig_schema_version) >= parse("2.0.0"):
            ephys_modules = []
            for data_stream in data_streams:
                ephys_modules = [
                    stream
                    for stream in data_stream["configurations"]
                    if stream["object_type"] == "Ephys assembly config"
                ]
            ephys_assemblies = [
                assembly
                for assembly in rig["components"]
                if assembly["object_type"] == "Ephys assembly"
            ]
            laser_assemblies = [
                assembly
                for assembly in rig["components"]
                if assembly["object_type"] == "Laser assembly"
            ]

            # gather all probes and lasers
            probe_devices = {}
            laser_devices = {}

            for ephys_assembly in ephys_assemblies:
                probes_in_assembly = ephys_assembly["probes"]

                for probe_info in probes_in_assembly:
                    probe_device_name = probe_info["name"]
                    probe_model_name = probe_info.get("probe_model", None)
                    probe_device_manufacturer = probe_info.get(
                        "manufacturer", None
                    )
                    if isinstance(probe_device_manufacturer, dict):
                        probe_device_manufacturer = (
                            probe_device_manufacturer.get("abbreviation")
                        )
                    probe_serial_number = probe_info.get("serial_number", None)
                    probe_device_description = ""
                    if probe_device_name is None:
                        if probe_model_name is not None:
                            probe_device_name = probe_model_name
                        else:
                            probe_device_name = "Probe"
                    if probe_model_name is not None:
                        probe_device_description += (
                            f"Model: {probe_device_description}"
                        )
                    if probe_serial_number is not None:
                        if len(probe_device_description) > 0:
                            probe_device_description += " - "
                        probe_device_description += (
                            f"Serial number: {probe_serial_number}"
                        )
                    probe_device = Device(
                        name=probe_device_name,
                        description=probe_device_description,
                        manufacturer=probe_device_manufacturer,
                    )
                    if probe_device_name not in probe_devices:
                        probe_devices[probe_device_name] = probe_device

            for laser_assembly in laser_assemblies:
                for laser in laser_assembly["lasers"]:
                    laser_device_name = laser["name"]
                    laser_device_description, laser_device_manufacturer = (
                        get_laser_description_manufacturer(laser, "external")
                    )
                    external_laser_device = Device(
                        name=laser_device_name,
                        description=laser_device_description,
                        manufacturer=laser_device_manufacturer,
                    )
                    if laser_device_name not in laser_devices:
                        laser_devices[laser_device_name] = (
                            external_laser_device
                        )

            # get probes and lasers used in the session
            devices = {}
            devices_target_location = {}
            for ephys_module in ephys_modules:
                for probe_name, probe_device in probe_devices.items():
                    if probe_name not in devices:
                        devices[probe_name] = probe_device
                        device_target_location = None
                        probe_configs = ephys_module["probes"]
                        for config in probe_configs:
                            primary_targeted_structure = config.get(
                                "primary_targeted_structure"
                            )
                            if primary_targeted_structure is not None:
                                if isinstance(
                                    primary_targeted_structure, dict
                                ):
                                    device_target_location = (
                                        primary_targeted_structure.get(
                                            "acronym"
                                        )
                                    )
                                else:
                                    device_target_location = (
                                        primary_targeted_structure
                                    )
                            devices_target_location[probe_name] = (
                                device_target_location
                            )

            if len(stimulus_device_names) > 0:
                for stimulus_device_name in stimulus_device_names:
                    if (
                        stimulus_device_name in laser_devices
                        and stimulus_device_name not in devices
                    ):
                        devices[stimulus_device_name] = laser_devices[
                            stimulus_device_name
                        ]
        else:
            warnings.warn(
                f"v{rig_schema_version} for instrument schema is "
                "not currently supported"
            )
    else:
        warnings.warn(f"Instrument file not found in {session_folder}")

    return devices, devices_target_location


def get_laser_description_manufacturer(laser, type) -> tuple[str, str]:
    """
    Gets the laser descrption and device manufacturer

    Parameters
    ----------
    laser: dict
        Information about laser metadata
    type: str
        Type for device description. Internal or External

    Returns
    -------
    tuple[str, str]
        The laser device description and manufacturer
    """
    laser_device_description = f"Type: {type} "
    wavelength = laser.get("wavelength", None)
    if wavelength is not None:
        laser_device_description += f" - Wavelength: {wavelength} "
        laser_device_description += \
            f"{laser.get('wavelength_unit', 'nanometer')}"
    max_power = laser.get("maximum_power", None)
    if max_power is not None:
        laser_device_description += (
            f" - Max power: {max_power} {laser.get('power_unit', 'milliwatt')}"
        )
    coupling = laser.get("coupling", None)
    if coupling is not None:
        laser_device_description += f" - Coupling: {coupling}"
    laser_device_manufacturer = laser.get("manufacturer", None)
    if isinstance(laser_device_manufacturer, dict):
        laser_device_manufacturer = laser_device_manufacturer.get("name", None)
    return laser_device_description, laser_device_manufacturer
