"""Example test template."""

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import h5py
import numpy as np
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.base import (  # example NWB container
    Images,
    ProcessingModule,
    VectorData,
)
from pynwb.epoch import TimeIntervals
from pynwb.event import EventsTable
from pynwb.file import Device, Subject

from aind_nwb_utils import NWBCombineIO
from aind_nwb_utils.nwb_io import determine_io
from aind_nwb_utils.utils import (
    _get_generation_code,
    _get_session_start_date_time,
    _handle_events_table,
    _handle_time_intervals,
    add_data,
    cast_timeseries_if_needed,
    cast_vectordata_if_needed,
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

    def test_add_data_to_processing(self):
        """Test adding data to processing"""
        nwbfile = create_autospec(NWBFile)
        obj = create_autospec(ProcessingModule)
        obj.name = "test_processing"

        # Simulate no pre-existing object with this name
        setattr(nwbfile, "processing", {})

        add_data(nwbfile, "processing", obj.name, obj)
        nwbfile.add_processing_module.assert_called_once_with(obj)

    def test_add_data_to_analysis(self):
        """Test adding data to analysis"""
        nwbfile = create_autospec(NWBFile)
        obj = create_autospec(
            Images
        )  # Using Images as an example analysis object
        obj.name = "test_analysis"

        # Simulate no pre-existing object with this name
        setattr(nwbfile, "analysis", {})

        add_data(nwbfile, "analysis", obj.name, obj)
        nwbfile.add_analysis.assert_called_once_with(obj)

    def test_add_data_to_intervals(self):
        """Test adding data to intervals"""
        nwbfile = create_autospec(NWBFile)
        obj = create_autospec(TimeIntervals)
        obj.name = "test_intervals"

        # Simulate no pre-existing object with this name
        setattr(nwbfile, "intervals", {})

        add_data(nwbfile, "intervals", obj.name, obj)
        nwbfile.add_time_intervals.assert_called_once_with(obj)

    def test_add_data_to_events(self):
        """Test adding data to events"""
        nwbfile = create_autospec(NWBFile)
        obj = create_autospec(EventsTable)
        obj.name = "test_events"

        # Simulate no pre-existing object with this name
        setattr(nwbfile, "events", {})

        nwbfile.add_events_table = MagicMock()

        add_data(nwbfile, "events", obj.name, obj)
        nwbfile.add_events_table.assert_called_once_with(obj)

    def test_get_nwb_attribute(self):
        """Test get_nwb_attribute function"""
        with NWBCombineIO(self.behavior_fp, [self.eye_tracking_fp]) as (
            result_nwb,
            _,
        ):
            eye_io = determine_io(self.eye_tracking_fp)
            with eye_io(self.eye_tracking_fp, "r") as io:
                eye_nwb = io.read()
            self.assertNotEqual(result_nwb, eye_nwb)

    def test_combine_nwb_file(self):
        """Test combine_nwb_file function"""
        with NWBCombineIO(self.behavior_fp, [self.eye_tracking_fp]) as (
            result,
            _,
        ):
            self.assertTrue(isinstance(result, NWBFile))

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

    def test_cast_timeseries_if_needed_exception_handling(self):
        """Test that exceptions during casting are handled gracefully"""
        from unittest.mock import patch

        # Create test data with float64 dtype
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Create TimeSeries with float64 data
        ts = TimeSeries(
            name="test_exception_timeseries",
            data=data,
            unit="volts",
            rate=1000.0,
            description="Test exception timeseries",
        )

        # Mock numpy asarray to raise an exception during casting
        with patch(
            "numpy.asarray", side_effect=Exception("Mock casting error")
        ):
            # Capture logger output to verify error message
            with patch("aind_nwb_utils.utils.logger.exception") as mock_logger:
                result = cast_timeseries_if_needed(ts)

                # Should return original object when casting fails
                self.assertEqual(id(result), id(ts))

                # Verify error message was logged
                mock_logger.assert_called_once_with(
                    "Could not cast TimeSeries 'test_exception_timeseries'"
                    " + Mock casting error"
                )

    def test_cast_vectordata_if_needed_no_casting_needed(self):
        """Test that VectorData with float32/int32 is returned unchanged"""
        # Create test data with float32 dtype (no casting needed)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Create VectorData with float32 data
        vector_data = VectorData(
            name="test_vectordata_float32",
            data=data,
            description="Test float32 vector data",
        )

        # Store original data reference
        original_data = vector_data.data

        # Cast the VectorData
        result = cast_vectordata_if_needed(vector_data)

        # Should return the same object
        self.assertEqual(id(result), id(vector_data))

        # Data should not be modified (same reference)
        self.assertEqual(id(result.data), id(original_data))
        self.assertEqual(result.data.dtype, np.float32)

    def test_cast_vectordata_if_needed_non_vectordata_object(self):
        """Test that non-VectorData objects are returned unchanged"""
        # Test with a string
        test_string = "not a vectordata"
        result = cast_vectordata_if_needed(test_string)
        self.assertEqual(result, test_string)

        # Test with a TimeSeries (different type)
        ts = TimeSeries(
            name="test_ts",
            data=np.array([1.0, 2.0]),
            unit="volts",
            rate=1000.0,
            description="Test timeseries",
        )
        result = cast_vectordata_if_needed(ts)
        self.assertEqual(id(result), id(ts))

    def test_cast_vectordata_if_needed_no_data_attribute(self):
        """Test VectorData without data attribute"""
        # Mock a VectorData-like object without data attribute
        mock_vectordata = MagicMock(spec=VectorData)
        mock_vectordata.name = "mock_vectordata"
        # Remove data attribute
        del mock_vectordata.data

        # Should return the original object unchanged
        result = cast_vectordata_if_needed(mock_vectordata)
        self.assertEqual(id(result), id(mock_vectordata))

    def test_cast_vectordata_if_needed_exception_handling(self):
        """Test that exceptions during VectorData casting are handled"""
        from unittest.mock import patch

        # Create test data with float64 dtype
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Create VectorData with float64 data
        vector_data = VectorData(
            name="test_exception_vectordata",
            data=data,
            description="Test exception vector data",
        )

        # Store original data for comparison
        original_data = vector_data.data

        # Mock numpy asarray to raise an exception during casting
        with patch(
            "numpy.asarray", side_effect=Exception("Mock casting error")
        ):
            # Capture logger output to verify error message
            with patch("aind_nwb_utils.utils.logger.exception") as mock_logger:
                result = cast_vectordata_if_needed(vector_data)

                # Should return original object when casting fails
                self.assertEqual(id(result), id(vector_data))

                # Data should remain unchanged
                np.testing.assert_array_equal(result.data, original_data)
                self.assertEqual(result.data.dtype, np.float64)

                # Verify error message was logged
                mock_logger.assert_called_once_with(
                    "Could not cast VectorData 'test_exception_vectordata' +"
                    " Mock casting error"
                )

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

    def test_get_subject_nwb_object_2_0(self):
        """Test get_subject_nwb_object with data schema 2.0 subject"""
        with open(Path("tests/resources/data_description.json"), "r") as f:
            data_description = json.load(f)

        with open(Path("tests/resources/subject_2_0.json"), "r") as f:
            subject_metadata = json.load(f)

        subject_object = get_subject_nwb_object(
            data_description, subject_metadata
        )
        self.assertTrue(isinstance(subject_object, Subject))

    def test_create_nwb_base_file(self):
        """Test create_nwb_base_file with ADS 1.x (session.json)"""
        import os

        with open(Path("tests/resources/data_description.json")) as f:
            data_description = json.load(f)
        with open(Path("tests/resources/session.json")) as f:
            session_metadata = json.load(f)
        project_name = data_description.get("project_name", "Unknown Project")
        session_type = session_metadata.get("session_type", "No specified")

        nwb_file_base = create_base_nwb_file(Path("tests/resources"))

        self.assertIsInstance(nwb_file_base, NWBFile)
        self.assertIn(project_name, nwb_file_base.session_description)
        self.assertIn(session_type, nwb_file_base.session_description)

        fd, nwb_path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            with NWBHDF5IO(nwb_path, "w") as nwb_io:
                nwb_io.write(nwb_file_base)
            with h5py.File(nwb_path, "r") as h5_file:
                self.assertEqual(
                    h5_file["general/was_generated_by"].shape,
                    (1, 2),
                )
        finally:
            os.unlink(nwb_path)

    @patch(
        "aind_nwb_utils.utils.package_version",
        return_value="0.2.7",
    )
    def test_get_generation_code(self, mock_package_version):
        """Test generation code across ADS processing metadata versions."""
        code = {"name": "producer", "version": "1.2.3"}
        ads_1 = {"processing_pipeline": {"data_processes": [{"code": code}]}}
        ads_2 = {"data_processes": [{"code": code}, {"code": None}]}

        expected = [
            ["aind-nwb-utils", "0.2.7"],
            ["producer", "1.2.3"],
        ]
        self.assertEqual(_get_generation_code(ads_1), expected)
        self.assertEqual(_get_generation_code(ads_2), expected)
        mock_package_version.assert_called_with("aind-nwb-utils")

    def test_create_nwb_base_file_ads2(self):
        """Test create_nwb_base_file with ADS 2.x (acquisition.json).

        Note: data_description/subject/procedures are ADS 1.x fixtures reused
        for convenience; only acquisition.json is ADS 2.x. The ADS 2.x
        fixture has no processing.json.
        """
        ads2_path = Path("tests/resources/ads2")
        with open(ads2_path / "acquisition.json") as f:
            acquisition = json.load(f)
        with open(Path("tests/resources/data_description.json")) as f:
            data_description = json.load(f)
        with open(Path("tests/resources/subject_2_0.json")) as f:
            subject = json.load(f)
        with open(Path("tests/resources/procedures.json")) as f:
            procedures = json.load(f)

        metadata_map = {
            ads2_path / "data_description.json": data_description,
            ads2_path / "subject.json": subject,
            ads2_path / "procedures.json": procedures,
            ads2_path / "acquisition.json": acquisition,
        }

        with patch(
            "aind_nwb_utils.utils.open_metadata_jsons",
            return_value=metadata_map,
        ):
            nwb_file_base = create_base_nwb_file(ads2_path)

        acquisition_type = acquisition.get("acquisition_type", "No specified")
        project_name = data_description.get("project_name", "Unknown Project")

        self.assertIsInstance(nwb_file_base, NWBFile)
        self.assertIsNotNone(nwb_file_base.session_start_time)
        self.assertIn(project_name, nwb_file_base.session_description)
        self.assertIn(acquisition_type, nwb_file_base.session_description)
        self.assertEqual(
            nwb_file_base.was_generated_by[0][0],
            "aind-nwb-utils",
        )

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

    def test_handle_time_intervals(self):
        """Test _handle_time_intervals function"""
        # Create a mock NWB file IO object
        mock_main_io = MagicMock()

        # Create a mock TimeIntervals object
        mock_time_intervals = MagicMock(spec=TimeIntervals)
        mock_time_intervals.name = "test_intervals"

        # Test with field_name == "intervals"
        _handle_time_intervals(mock_main_io, mock_time_intervals, "intervals")

        # Verify that reset_parent was called
        mock_time_intervals.reset_parent.assert_called_once()

        # Verify that parent was set to main_io
        self.assertEqual(mock_time_intervals.parent, mock_main_io)

        mock_main_io.add_time_intervals.assert_called_once_with(
            mock_time_intervals
        )

        # Reset mocks for next test
        mock_main_io.reset_mock()
        mock_time_intervals.reset_mock()

        # Test with field_name != "intervals" — still routed through intervals
        _handle_time_intervals(
            mock_main_io, mock_time_intervals, "other_field"
        )

        # Verify that reset_parent and parent setting still occur
        mock_time_intervals.reset_parent.assert_called_once()
        self.assertEqual(mock_time_intervals.parent, mock_main_io)

        mock_main_io.add_time_intervals.assert_called_once_with(
            mock_time_intervals
        )

    def test_handle_time_intervals_with_real_objects(self):
        """Test _handle_time_intervals with real NWB objects"""
        # Create a real NWB file for testing
        nwb_file = NWBFile(
            session_description="test session",
            identifier="test_id",
            session_start_time=datetime.datetime.now(tz=datetime.timezone.utc),
        )

        # Create a TimeIntervals object
        time_intervals = TimeIntervals(
            name="test_intervals",
            description="Test time intervals for testing",
        )

        # Add some sample intervals
        time_intervals.add_interval(
            start_time=0.0, stop_time=1.0, tags=["test_tag"], timeseries=[]
        )

        # Test the function
        _handle_time_intervals(nwb_file, time_intervals, "intervals")

        # Verify the time intervals were added to the NWB file
        self.assertIn("test_intervals", nwb_file.intervals)
        self.assertEqual(nwb_file.intervals["test_intervals"], time_intervals)
        self.assertEqual(time_intervals.parent, nwb_file)

    def test_handle_events_table_new_table(self):
        """Test _handle_events_table function with new events table"""
        # Create a mock NWB file IO object
        mock_main_io = MagicMock()
        mock_main_io.events = {}  # No existing events

        # Create a mock EventsTable object
        mock_events_table = MagicMock(spec=EventsTable)
        mock_events_table.name = "test_events"

        # Test adding new events table
        _handle_events_table(mock_main_io, mock_events_table, "test_events")

        # Verify that add_events_table was called for new table
        mock_main_io.add_events_table.assert_called_once_with(
            mock_events_table
        )

    def test_handle_events_table_merge_existing_new_columns(self):
        """Test _handle_events_table merging with existing table"""
        # Create a mock NWB file IO object
        mock_main_io = MagicMock()

        # Create mock existing events table with a column object
        mock_existing_col = MagicMock()
        mock_existing_col.name = "existing_col"
        mock_existing_table = MagicMock()
        mock_existing_table.columns = [mock_existing_col]
        mock_main_io.events = {"test_events": mock_existing_table}

        # Create a mock EventsTable with a new column object
        mock_new_column = MagicMock()
        mock_new_column.name = "new_col"
        mock_new_events_table = MagicMock(spec=EventsTable)
        mock_new_events_table.name = "test_events"
        mock_new_events_table.columns = [mock_new_column]

        # Test merging with existing table
        _handle_events_table(
            mock_main_io, mock_new_events_table, "test_events"
        )

        mock_existing_table.add_column.assert_called_once_with(mock_new_column)
        mock_main_io.add_events_table.assert_not_called()

    def test_handle_events_table_merge_existing_extend_data(self):
        """Test _handle_events_table extend existing columns"""
        # Create a mock NWB file IO object
        mock_main_io = MagicMock()

        # Create mock existing events table with a column object
        mock_existing_column = MagicMock()
        mock_existing_column.name = "shared_col"
        mock_existing_column.data = MagicMock()  # Mock the data object itself
        mock_existing_column.data.extend = (
            MagicMock()
        )  # Mock the extend method
        mock_existing_table = MagicMock()
        mock_existing_table.columns = [mock_existing_column]
        mock_existing_table.__getitem__.return_value = mock_existing_column
        mock_main_io.events = {"test_events": mock_existing_table}

        # Create a mock EventsTable with same column name
        mock_new_column = MagicMock()
        mock_new_column.name = "shared_col"
        mock_new_column.data = [4, 5, 6]
        mock_new_events_table = MagicMock(spec=EventsTable)
        mock_new_events_table.name = "test_events"
        mock_new_events_table.columns = [mock_new_column]

        # Test merging with existing table
        _handle_events_table(
            mock_main_io, mock_new_events_table, "test_events"
        )

        # Verify that data was extended
        mock_existing_column.data.extend.assert_called_once_with([4, 5, 6])

        # Verify that add_column was NOT called (column already exists)
        mock_existing_table.add_column.assert_not_called()

        # Verify that add_events_table was NOT called (we're merging)
        mock_main_io.add_events_table.assert_not_called()

    def test_handle_events_table_merge_mixed_columns(self):
        """Test _handle_events_table with both new and existing columns"""
        # Create a mock NWB file IO object
        mock_main_io = MagicMock()

        # Create mock existing events table with a column object
        mock_existing_col_obj = MagicMock()
        mock_existing_col_obj.name = "existing_col"
        mock_existing_column = MagicMock()
        mock_existing_column.data = MagicMock()  # Mock the data object itself
        mock_existing_column.data.extend = (
            MagicMock()
        )  # Mock the extend method
        mock_existing_table = MagicMock()
        mock_existing_table.columns = [mock_existing_col_obj]
        mock_existing_table.__getitem__.return_value = mock_existing_column
        mock_main_io.events = {"test_events": mock_existing_table}

        # Create a mock EventsTable with both existing and new columns
        mock_new_events_table = MagicMock(spec=EventsTable)
        mock_new_events_table.name = "test_events"

        # Mock columns as a list of column objects with .name
        mock_existing_column_new_data = MagicMock()
        mock_existing_column_new_data.name = "existing_col"
        mock_existing_column_new_data.data = [4, 5, 6]
        mock_new_column = MagicMock()
        mock_new_column.name = "new_col"
        mock_new_events_table.columns = [
            mock_existing_column_new_data,
            mock_new_column,
        ]

        # Test merging with existing table
        _handle_events_table(
            mock_main_io, mock_new_events_table, "test_events"
        )

        # Verify that data was extended for existing column
        mock_existing_column.data.extend.assert_called_once_with([4, 5, 6])

        # Verify that add_column was called for new column
        mock_existing_table.add_column.assert_called_once_with(mock_new_column)

        # Verify that add_events_table was NOT called
        mock_main_io.add_events_table.assert_not_called()

    def test_combine_nwb_write_zarr(self):
        """Test NWBCombineIO.write with zarr format"""
        import shutil
        import tempfile

        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / "combined_output"
        try:
            combiner = NWBCombineIO(self.behavior_fp, [self.eye_tracking_fp])
            combiner.read()
            combiner.write(output_path, format="zarr")
            combiner.close()
            self.assertTrue(output_path.exists())
        finally:
            shutil.rmtree(output_dir)

    def test_combine_nwb_write_hdf5(self):
        """Test NWBCombineIO.write with hdf5 format"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".nwb", delete=False) as tmp:
            output_path = Path(tmp.name)
        try:
            combiner = NWBCombineIO(self.behavior_fp, [self.eye_tracking_fp])
            combiner.read()
            combiner.write(output_path, format="hdf5")
            combiner.close()
            self.assertTrue(output_path.exists())
        finally:
            output_path.unlink(missing_ok=True)

    def test_combine_nwb_write_invalid_format(self):
        """Test NWBCombineIO.write raises ValueError for invalid format"""
        combiner = NWBCombineIO(self.behavior_fp, [self.eye_tracking_fp])
        combiner.read()
        with self.assertRaises(ValueError):
            combiner.write(Path("output"), format="parquet")
        combiner.close()


if __name__ == "__main__":
    unittest.main()
