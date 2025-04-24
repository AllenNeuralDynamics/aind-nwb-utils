"""Tests for nwb_io.py"""

import unittest
from pathlib import Path

from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
import tempfile

from aind_nwb_utils.nwb_io import create_temp_nwb, determine_io


class TestNWBIO(unittest.TestCase):
    """Tests for create_temp_nwb function"""

    def test_create_temp_nwb_with_nwbzarrio(self):
        """Test create_temp_nwb with NWBZarrIO"""
        temp_path = create_temp_nwb(None, NWBZarrIO)
        self.assertTrue(temp_path.is_dir())
        self.assertTrue(temp_path.exists())

    def test_create_temp_nwb_with_nwbhdf5io(self):
        """Test create_temp_nwb with NWBHDF5IO"""
        temp_path = create_temp_nwb(None, NWBHDF5IO)
        self.assertTrue(temp_path.is_file())
        self.assertTrue(temp_path.exists())

    def test_determine_io_with_directory(self):
        """Test determine_io with a directory path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            io_type = determine_io(temp_path)
            self.assertEqual(io_type, NWBZarrIO)

    def test_determine_io_with_file(self):
        """Test determine_io with a file path"""
        with tempfile.NamedTemporaryFile(suffix=".nwb") as temp_file:
            temp_path = Path(temp_file.name)
            io_type = determine_io(temp_path)
            self.assertEqual(io_type, NWBHDF5IO)


if __name__ == "__main__":
    unittest.main()
