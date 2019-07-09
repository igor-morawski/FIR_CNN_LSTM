import unittest

import os
import numpy as np
from tools import dataset

test_data_dir = os.path.join("tests", "data")
test_sequence_fn = os.path.join('tests', 'data', '20170203', 'raw', '20170203_p1_dark1.csv')

#tests\\data\\20170203\\raw\\20170203_p1_dark1.csv
test_fn1 = os.path.join('tests', 'data', '20170203', 'raw', '20170203_p1_dark1.csv')
#tests\\data\\20170210\\raw\\20170210_p1_dark1.csv
test_fn2 = os.path.join('tests', 'data', '20170210', 'raw', '20170210_p1_dark1.csv')

test_array_fn = os.path.join("tests", "files", "array.csv")
test_array = np.hstack([np.arange(256), np.arange(256)*2, np.arange(256)*4]).reshape([3,16,16]).astype(dtype=np.float64)
test_array_minmax_normalized = test_array/2040

class TestDataset(unittest.TestCase):
    def test_no_directory(self):
        with self.assertRaises(OSError):
            no_data = dataset.Dataset(test_data_dir+"nosuchdirectory")

    def test_attributes(self):
        data = dataset.Dataset(test_data_dir)
        self.assertEqual(len(data), 2)
        self.assertEqual(len(data.annotation), 1400)
        actors = np.array(['human1', 'human2', 'human7', 'human8', 'human9'], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        self.assertEqual(data.minmax_normalized, None)
    
    def test_normalizing(self):
        data = dataset.Dataset(test_data_dir, minmax_normalized=True)
        self.assertEqual(data.minmax_normalized, True)
        self.assertEqual(data[0].temp_min, 0)
        self.assertEqual(data[0].temp_max, 1)

    
    def test_sampling(self):
        samples = dataset.Dataset(test_data_dir, sample=True, samples_k=1)
        self.assertEqual(len(samples), 1)

    def test_actors_filtering(self):
        data = dataset.Dataset(test_data_dir)
        actors = np.array(['human1', 'human2', 'human7', 'human8', 'human9'], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = [test_fn1, test_fn2]
        self.assertEqual(data.sequences, sequences)

        data.excludeActor('human8')
        actors = np.array(['human1', 'human2', 'human7', 'human9'], dtype=object)
        sequences = [test_fn1, test_fn2]
        self.assertEqual(data.sequences, sequences)
        self.assertTrue(np.array_equal(data.actors, actors))

        data.filterActor('human7')
        actors = np.array(['human7'], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = [test_fn2]
        self.assertEqual(data.sequences, sequences)

        data.filterActor('shouldnowbeempty')
        actors = np.array([], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = []
        self.assertEqual(data.sequences, sequences)

        data = dataset.Dataset(test_data_dir, actor = 'human7')
        actors = np.array(['human7'], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = [test_fn2]
        self.assertEqual(data.sequences, sequences)

        data = dataset.Dataset(test_data_dir, exclude_actor = 'human1')
        data.excludeActor('human2')
        actors = np.array(['human7', 'human8', 'human9'])
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = [test_fn2]
        self.assertEqual(data.sequences, sequences)

    def test_reloading_actors(self):
        data = dataset.Dataset(test_data_dir)
        actors = np.array(['human1', 'human2', 'human7', 'human8', 'human9'], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = [test_fn1, test_fn2]
        self.assertEqual(data.sequences, sequences)

        data.excludeActor('human8')
        actors = np.array(['human1', 'human2', 'human7', 'human9'], dtype=object)
        sequences = [test_fn1, test_fn2]
        self.assertEqual(data.sequences, sequences)
        self.assertTrue(np.array_equal(data.actors, actors))

        data.filterActor('human7')
        actors = np.array(['human7'], dtype=object)
        self.assertTrue(np.array_equal(data.actors, actors))
        sequences = [test_fn2]
        self.assertEqual(data.sequences, sequences)

        data.reload() 
        actors = np.array(['human1', 'human2', 'human7', 'human8', 'human9'], dtype=object)
        sequences = [test_fn1, test_fn2]
        self.assertEqual(data.sequences, sequences)
        self.assertTrue(np.array_equal(data.actors, actors))



class TestSequence(unittest.TestCase):
    def test_attributes(self):
        sample = dataset.Sequence(test_sequence_fn)
        self.assertEqual(len(sample), 398)
        self.assertEqual(sample.filename, test_sequence_fn)
        self.assertEqual(sample.sequence_name, "20170203_p1_dark1.csv")
    
    def test_frame_range(self):
        sample = dataset.Sequence(test_sequence_fn, frame_start = 10, frame_stop=20)
        self.assertEqual(len(sample), 10)
        sample = dataset.Sequence(test_sequence_fn, frame_start = 10, frame_stop=11)
        self.assertEqual(len(sample), 1)

    def test_frame_start(self):
        sample = dataset.Sequence(test_sequence_fn, frame_start = 10)
        self.assertEqual(len(sample), 388)
    def test_min_max(self):
        if dataset.SKIP_FRAMES != 20:
            self.skipTest("SKIP_FRAMES must be set to 20")
        else:
            sample = dataset.Sequence(test_sequence_fn)
            self.assertAlmostEqual(sample.temp_min, 23.375)
            self.assertAlmostEqual(sample.temp_max, 25.165625)
    
    def test_example(self):
        if dataset.SKIP_FRAMES != 20:
            self.skipTest("SKIP_FRAMES must be set to 20")
        frame_start = 20
        frame_stop = 23
        sequence = dataset.Sequence(test_array_fn, frame_start = frame_start, frame_stop = frame_stop)
        self.assertEqual(sequence.shape, (3, 16, 16))
        self.assertEqual(sequence.temp_min, 0)
        self.assertEqual(sequence.temp_max, 2040)
        self.assertEqual(sequence.start, frame_start)
        self.assertEqual(sequence.stop, frame_stop)
        self.assertTrue(np.array_equal(sequence, test_array))

    def test_minmax_normalizing(self):
        if dataset.SKIP_FRAMES != 20:
            self.skipTest("SKIP_FRAMES must be set to 20")
        frame_start = 20
        frame_stop = 23
        sequence = dataset.Sequence(test_array_fn, frame_start = frame_start, frame_stop = frame_stop, minmax_normalized=False)
        self.assertTrue(np.array_equal(sequence, test_array))
        self.assertEqual(sequence.temp_min, 0)
        self.assertEqual(sequence.temp_max, 2040)
        sequence = dataset.Sequence(test_array_fn, frame_start = frame_start, frame_stop = frame_stop, minmax_normalized=True)
        self.assertTrue(np.array_equal(sequence, test_array_minmax_normalized))
        self.assertEqual(sequence.temp_min, 0)
        self.assertEqual(sequence.temp_max, 1)

        frame_start = 23
        frame_stop = 24
        sequence = dataset.Sequence(test_array_fn, frame_start = frame_start, frame_stop = frame_stop, minmax_normalized=False)
        self.assertTrue(np.array_equal(sequence, 2040*np.ones([1,16,16])))
        self.assertEqual(sequence.temp_min, 0)
        self.assertEqual(sequence.temp_max, 2040)

        sequence = dataset.Sequence(test_array_fn, frame_start = frame_start, frame_stop = frame_stop, minmax_normalized=True)
        self.assertTrue(np.array_equal(sequence, np.ones([1,16,16])))
        self.assertEqual(sequence.temp_min, 0)
        self.assertEqual(sequence.temp_max, 1)

    def test_as_uint8(self):
        frame_start = 20
        frame_stop = None
        sequence_notnormalized = dataset.Sequence(test_fn1, frame_start = frame_start, frame_stop = frame_stop, minmax_normalized=False)
        sequence_normalized = dataset.Sequence(test_fn1, frame_start = frame_start, frame_stop = frame_stop, minmax_normalized=True)
        self.assertNotEqual(sequence_notnormalized.temp_min, 0)
        self.assertNotEqual(sequence_notnormalized.temp_max, 1)
        self.assertEqual(sequence_normalized.temp_min, 0)
        self.assertEqual(sequence_normalized.temp_max, 1)
        self.assertFalse(np.array_equal(sequence_notnormalized, sequence_normalized))
        normalized_uint8 = (255 * sequence_normalized).astype(np.uint8)
        self.assertTrue(np.array_equal(sequence_notnormalized.as_uint8(), normalized_uint8))

        

