"""Example test template."""

import unittest
from pathlib import Path

from pynwb import NWBHDF5IO

from aind_nwb_utils.nwb_io import determine_io
from aind_nwb_utils.utils import combine_nwb_file


class TestUtils(unittest.TestCase):
    """Tests for utils.py"""

    @classmethod
    def setUp(cls):
        """Set up the test class"""
        cls.eye_tracking_fp = Path(
            "tests/resources/multiplane-ophys_eye-tracking"
        )
        cls.behavior_fp = Path("tests/resources/multiplan-ophys_behavior.nwb")

    def test_add_nwb_attribute(self):
        """Test add_nwb_attribute function"""
        result = combine_nwb_file(
            self.behavior_fp, self.eye_tracking_fp, NWBHDF5IO
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
            Path(self.eye_tracking_fp), Path(self.behavior_fp), NWBHDF5IO
        )
        self.assertTrue(result_fp.exists())


if __name__ == "__main__":
    unittest.main()
