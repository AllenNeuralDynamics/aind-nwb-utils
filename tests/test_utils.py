"""Example test template."""

import datetime
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

from pynwb import NWBHDF5IO, NWBFile
from pynwb.base import Images  # example NWB container
from pynwb.file import Device, Subject

from aind_nwb_utils.nwb_io import determine_io
from aind_nwb_utils.utils import (
    _get_session_start_date_time,
    add_data,
    combine_nwb_file,
    create_base_nwb_file,
    get_ephys_devices_from_metadata,
    get_subject_nwb_object,
    is_non_mergeable,
)


class TestUtils(unittest.TestCase):
    """Tests for utils.py"""

    @classmethod
    def setUp(cls):
        """Set up the test class"""
        cls.eye_tracking_fp = Path(
            "tests/resources/multiplane-ophys_eye-tracking"
        )
        cls.behavior_fp = Path("tests/resources/multiplan-ophys_behavior.nwb")

    def test_is_non_mergeable_false(self):
        """Should return False for mergeable/custom container types"""
        self.assertFalse(
            is_non_mergeable(NWBFile("desc", "id", datetime.datetime.now()))
        )

    def test_is_non_mergeable_various_types(self):
        """Should return True for non-mergeable types"""
        self.assertTrue(is_non_mergeable("string"))
        self.assertTrue(is_non_mergeable(datetime.datetime.now()))
        self.assertTrue(is_non_mergeable([]))

    def test_add_data_to_acquisition(self):
        """Test adding data to acquisition"""
        nwbfile = create_autospec(NWBFile)
        obj = create_autospec(Images)
        obj.name = "test_image"

        # Simulate no pre-existing object with this name
        setattr(nwbfile, "acquisition", {})

        add_data(nwbfile, "acquisition", obj.name, obj)
        nwbfile.add_acquisition.assert_called_once_with(obj)

    def test_add_data_with_existing_name(self):
        """Should return early if name already exists"""
        nwbfile = MagicMock()
        nwbfile.acquisition = {"existing": "dummy"}
        obj = MagicMock()
        obj.name = "existing"

        # Should return without calling add_acquisition
        add_data(nwbfile, "acquisition", obj.name, obj)
        nwbfile.add_acquisition.assert_not_called()

    def test_add_data_with_unknown_field_raises(self):
        """Should raise ValueError for unknown field"""
        nwbfile = MagicMock()
        obj = MagicMock()
        obj.name = "anything"
        with self.assertRaises(ValueError):
            add_data(nwbfile, "unknown", obj.name, obj)

    def test_get_nwb_attribute(self):
        """Test get_nwb_attribute function"""
        result = combine_nwb_file(
            self.behavior_fp, self.eye_tracking_fp, None, NWBHDF5IO
        )
        result_io = determine_io(result)
        with result_io(result, "r") as io:
            result_nwb = io.read()
        eye_io = determine_io(self.eye_tracking_fp)
        with eye_io(self.eye_tracking_fp, "r") as io:
            eye_nwb = io.read()
        self.assertNotEqual(result_nwb, eye_nwb)

    def test_combine_nwb_file(self):
        """Test combine_nwb_file function"""
        result_fp = combine_nwb_file(
            Path(self.eye_tracking_fp), Path(self.behavior_fp), None, NWBHDF5IO
        )
        self.assertTrue(result_fp.exists())

    def test_get_session_start_date_time(self):
        """Test _get_session_start_date_time"""
        with open(Path("tests/resources/data_description.json"), "r") as f:
            data_description = json.load(f)

        session_start_date_time = _get_session_start_date_time(
            data_description["creation_time"]
        )
        self.assertTrue(isinstance(session_start_date_time, datetime.datetime))

    def test_get_subject_nwb_object(self):
        """Test get_subject_nwb_object"""
        with open(Path("tests/resources/data_description.json"), "r") as f:
            data_description = json.load(f)

        with open(Path("tests/resources/subject.json"), "r") as f:
            subject_metadata = json.load(f)

        subject_object = get_subject_nwb_object(
            data_description, subject_metadata
        )
        self.assertTrue(isinstance(subject_object, Subject))

    def test_create_nwb_base_file(self):
        """Test create_nwb_base_file"""
        nwb_file_base = create_base_nwb_file(Path("tests/resources"))
        self.assertTrue(isinstance(nwb_file_base, NWBFile))

    def test_get_ephys_devices_from_metadata_ads2(self):
        """Test get_ephys_devices_from_metadata with aind-data-schema v2.x"""
        devices, devices_target_location = get_ephys_devices_from_metadata(
            "tests/resources/ads2"
        )
        self.assertIsInstance(devices, dict)
        self.assertIsInstance(devices_target_location, dict)
        self.assertTrue(devices.keys())
        self.assertTrue(devices_target_location.keys())
        self.assertIsInstance(devices["Probe A"], Device)
        self.assertEqual(devices_target_location["Probe A"], "LGd")

    def test_get_ephys_devices_from_metadata_ads1(self):
        """Test get_ephys_devices_from_metadata with aind-data-schema v1.x"""
        devices, devices_target_location = get_ephys_devices_from_metadata(
            "tests/resources/ads1"
        )
        self.assertIsInstance(devices, dict)
        self.assertIsInstance(devices_target_location, dict)
        self.assertTrue(devices.keys())
        self.assertTrue(devices_target_location.keys())
        self.assertIsInstance(devices["Probe A"], Device)
        self.assertEqual(devices_target_location["Probe A"], "ACB")


if __name__ == "__main__":
    unittest.main()
