import unittest

from FIR_CNN_LSTM.tools import dataset

test_data_dir = "test_dataset"

class TestDataset(unittest.TestCase):
    def test_no_directory(self):
        data = dataset.Dataset(test_data_dir+"nosuchdirectory")
        self.assertRaises(OSError(2, 'No such file or directory'))