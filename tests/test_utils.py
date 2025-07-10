"""Example test template."""

import unittest
import datetime
from pathlib import Path

from pynwb import NWBHDF5IO
from pynwb import NWBFile
from pynwb.base import Images  # example NWB container

from unittest.mock import create_autospec, MagicMock

from aind_nwb_utils.nwb_io import determine_io
from aind_nwb_utils.utils import (
    combine_nwb_file,
    is_non_mergeable,
    add_data,
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
            self.behavior_fp, self.eye_tracking_fp, "/test.nwb", NWBHDF5IO
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
            Path(self.eye_tracking_fp), Path(self.behavior_fp), "/test.nwb", NWBHDF5IO
        )
        self.assertTrue(result_fp.exists())


if __name__ == "__main__":
    unittest.main()
