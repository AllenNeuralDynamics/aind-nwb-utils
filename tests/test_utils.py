"""Example test template."""

import datetime
import tempfile
from pathlib import Path
import numpy as np

from pynwb import NWBHDF5IO
from pynwb import NWBFile
from pynwb import TimeSeries
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
    cast_timeseries_if_needed,
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
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.nwb"
            result = combine_nwb_file(
                self.behavior_fp, self.eye_tracking_fp, output_path, NWBHDF5IO
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
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.nwb"
            result_fp = combine_nwb_file(
                self.behavior_fp, self.eye_tracking_fp, output_path, NWBHDF5IO
            )
            self.assertTrue(result_fp.exists())

    def test_cast_timeseries_if_needed_float64_to_float32(self):
        """Test casting float64 TimeSeries data to float32"""
        # Create test data with float64 dtype
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Create TimeSeries with float64 data
        ts = TimeSeries(
            name="test_timeseries",
            data=data,
            unit="volts",
            rate=1000.0,
            description="Test timeseries",
        )

        # Cast the TimeSeries
        result = cast_timeseries_if_needed(ts)

        # Verify the result is a new TimeSeries object
        self.assertIsInstance(result, TimeSeries)
        self.assertNotEqual(id(ts), id(result))

        # Verify the data was cast to float32
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(result.name, "test_timeseries")
        self.assertEqual(result.unit, "volts")
        self.assertEqual(result.rate, 1000.0)
        self.assertEqual(result.description, "Test timeseries")

        # Verify data values are preserved
        np.testing.assert_array_equal(result.data, data.astype(np.float32))

    def test_cast_timeseries_if_needed_int64_to_int32(self):
        """Test casting int64 TimeSeries data to int32"""
        # Create test data with int64 dtype
        data = np.array([1, 2, 3], dtype=np.int64)

        # Create TimeSeries with int64 data
        ts = TimeSeries(
            name="test_timeseries_int",
            data=data,
            unit="counts",
            rate=500.0,
            description="Test int timeseries",
        )

        # Cast the TimeSeries
        result = cast_timeseries_if_needed(ts)

        # Verify the result is a new TimeSeries object
        self.assertIsInstance(result, TimeSeries)
        self.assertNotEqual(id(ts), id(result))

        # Verify the data was cast to int32
        self.assertEqual(result.data.dtype, np.int32)
        self.assertEqual(result.name, "test_timeseries_int")
        self.assertEqual(result.unit, "counts")
        self.assertEqual(result.rate, 500.0)
        self.assertEqual(result.description, "Test int timeseries")

        # Verify data values are preserved
        np.testing.assert_array_equal(result.data, data.astype(np.int32))

    def test_cast_timeseries_if_needed_no_casting_needed(self):
        """Test that TimeSeries with float32/int32 is returned unchanged"""
        # Create test data with float32 dtype (no casting needed)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Create TimeSeries with float32 data
        ts = TimeSeries(
            name="test_timeseries_float32",
            data=data,
            unit="volts",
            rate=1000.0,
            description="Test float32 timeseries",
        )

        # Cast the TimeSeries
        result = cast_timeseries_if_needed(ts)

        # Verify the original object is returned (no casting needed)
        self.assertEqual(id(ts), id(result))
        self.assertEqual(result.data.dtype, np.float32)

    def test_cast_timeseries_if_needed_non_timeseries_object(self):
        """Test that non-TimeSeries objects are returned unchanged"""
        # Test with a string
        test_string = "not a timeseries"
        result = cast_timeseries_if_needed(test_string)
        self.assertEqual(result, test_string)

        # Test with a list
        test_list = [1, 2, 3]
        result = cast_timeseries_if_needed(test_list)
        self.assertEqual(result, test_list)

        # Test with None
        result = cast_timeseries_if_needed(None)
        self.assertIsNone(result)

    def test_cast_timeseries_if_needed_with_all_parameters(self):
        """Test casting preserves all TimeSeries parameters"""
        # Create test data with float64 dtype
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        timestamps = np.array([0.0, 1.0, 2.0])

        # Create TimeSeries with all parameters
        ts = TimeSeries(
            name="test_full_timeseries",
            data=data,
            unit="volts",
            conversion=2.0,
            resolution=0.001,
            timestamps=timestamps,
            description="Full test timeseries",
            comments="Test comments",
            control=[0, 1, 2],
            control_description=["a", "b", "c"],
        )

        # Cast the TimeSeries
        result = cast_timeseries_if_needed(ts)

        # Verify all parameters are preserved
        self.assertEqual(result.name, "test_full_timeseries")
        self.assertEqual(result.unit, "volts")
        self.assertEqual(result.conversion, 2.0)
        self.assertEqual(result.resolution, 0.001)
        np.testing.assert_array_equal(result.timestamps, timestamps)
        self.assertEqual(result.description, "Full test timeseries")
        self.assertEqual(result.comments, "Test comments")
        self.assertEqual(result.control, [0, 1, 2])
        self.assertEqual(result.control_description, ["a", "b", "c"])

        # Verify data was cast to float32
        self.assertEqual(result.data.dtype, np.float32)
        np.testing.assert_array_equal(result.data, data.astype(np.float32))

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
