from tools import dataset
from tools.dataset import Dataset
from tools import prepare

import os
import argparse

import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_dir',
      type=str,
      default=os.path.join("..", "dataset"),
      help='Path to folder containing the FIR dataset.'
  )
  parser.add_argument(
    '--model_dir',
    type=str,
    default="/"+os.path.join("tmps", "model"),
    help='Where to save the trained model.'
)
  parser.add_argument(
    '--temperature_dir',
    type=str,
    default="/"+os.path.join("tmps", "cache", "temperature"),
    help='Where to save the cached sequences (temperature).'
)
  parser.add_argument(
    '--flow_dir',
    type=str,
    default="/"+os.path.join("tmps", "cache", "optical_flow"),
    help='Where to save the cached sequences (optical flow).'
)
  parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='How many epochs to run before ending.'
)
  parser.add_argument(
    '--frames',
    type=int,
    default=10,
    help='How many frames (time steps) in each unit.'
)
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='How large a learning rate to use when training.'
)
  parser.add_argument(
    '--train_batch_size',
    type=int,
    default=100,
    help='How many images to train on at a time.'
)
  parser.add_argument(
    "--download", action="store_true",
                      help='Download the dataset.'
)
  parser.add_argument(
    "--prepare", action="store_true",
                      help='Prepare the dataset.'
)
  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.download:
      dataset.download("..")
  
  data_normalized = Dataset(FLAGS.dataset_dir, minmax_normalized=True)

  if FLAGS.prepare:
    prepare.sequences_by_actor(data_normalized, FLAGS.temperature_dir)
    prepare.optical_flow(data_normalized, FLAGS.flow_dir)

