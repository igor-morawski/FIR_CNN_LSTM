import unittest

import os
import numpy as np

from tools import prepare
from tools.dataset import Dataset

test_data_dir = os.path.join("tests", "data")
test_sequence_fn = os.path.join('tests', 'data', '20170203', 'raw', '20170203_p1_dark1.csv')
#tests\\data\\20170203\\raw\\20170203_p1_dark1.csv
test_fn1 = os.path.join('tests', 'data', '20170203', 'raw', '20170203_p1_dark1.csv')
#tests\\data\\20170210\\raw\\20170210_p1_dark1.csv
test_fn2 = os.path.join('tests', 'data', '20170210', 'raw', '20170210_p1_dark1.csv')
test_cache_dir = os.path.join("tests", "tmp", "cache")
test_temperature_dir = os.path.join(test_cache_dir, "temperature")
test_flow_dir = os.path.join(test_cache_dir, "optical_flow")

class TestSequencesByActor(unittest.TestCase):
    def test(self):
        data_normalized = Dataset(test_data_dir, minmax_normalized=True)
        prepare.sequences_by_actor(data_normalized, test_temperature_dir)
        actors = np.array(['human1', 'human2', 'human7', 'human8', 'human9'], dtype=object)
        for actor in actors:
            self.assertTrue(os.path.exists(os.path.join(test_temperature_dir,actor)))
        actors = np.array(['human3', 'human4', 'human5', 'human6'], dtype=object)
        for actor in actors:
            self.assertFalse(os.path.exists(os.path.join(test_temperature_dir,actor)))
        files = []
        files.append(os.path.join(test_temperature_dir, "human1", "walk_20170203_p1_dark1_237_250.npy"))
        files.append(os.path.join(test_temperature_dir, "human2", "walk_20170203_p1_dark1_354_365.npy"))
        files.append(os.path.join(test_temperature_dir, "human7", "walk_20170210_p1_dark1_137_149.npy"))
        files.append(os.path.join(test_temperature_dir, "human8", "walk_20170210_p1_dark1_198_211.npy"))
        files.append(os.path.join(test_temperature_dir, "human9", "walk_20170210_p1_dark1_280_293.npy"))
        for fn in files:
            self.assertTrue(os.path.exists(fn))
        #files[0] is walk_20170203_p1_dark1_237_250
        frame_number = 250-237
        temperature = np.load(files[0])
        self.assertEqual(len(temperature), frame_number)
        #files[1] is walk_20170203_p1_dark1_354_365
        frame_number = 365-354
        temperature = np.load(files[1])
        self.assertEqual(len(temperature), frame_number)





class TestOpticalFlow(unittest.TestCase):
    def test(self):
        data_normalized = Dataset(test_data_dir, minmax_normalized=True)
        prepare.optical_flow(data_normalized, test_flow_dir)
        actors = np.array(['human1', 'human2', 'human7', 'human8', 'human9'], dtype=object)
        for actor in actors:
            self.assertTrue(os.path.exists(os.path.join(test_flow_dir,actor)))
        actors = np.array(['human3', 'human4', 'human5', 'human6'], dtype=object)
        for actor in actors:
            self.assertFalse(os.path.exists(os.path.join(test_flow_dir,actor)))
        files = []
        files.append(os.path.join(test_flow_dir, "human1", "walk_20170203_p1_dark1_237_250.npy"))
        files.append(os.path.join(test_flow_dir, "human2", "walk_20170203_p1_dark1_354_365.npy"))
        files.append(os.path.join(test_flow_dir, "human7", "walk_20170210_p1_dark1_137_149.npy"))
        files.append(os.path.join(test_flow_dir, "human8", "walk_20170210_p1_dark1_198_211.npy"))
        files.append(os.path.join(test_flow_dir, "human9", "walk_20170210_p1_dark1_280_293.npy"))
        for fn in files:
            self.assertTrue(os.path.exists(fn))
        #files[0] is walk_20170203_p1_dark1_237_250
        frame_number = 250-237-1
        flow = np.load(files[0])
        self.assertEqual(len(flow), frame_number)
        #files[1] is walk_20170203_p1_dark1_354_365
        frame_number = 365-354-1
        flow = np.load(files[1])
        self.assertEqual(len(flow), frame_number)