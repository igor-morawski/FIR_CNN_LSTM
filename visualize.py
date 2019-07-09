'''
Visualize sequences prepared by tools.prepare
Run after running main.py
'''

from tools import dataset
from tools.dataset import Dataset
from tools import prepare

import random

import os
import argparse

import numpy as np

from glob import glob

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--visualize_dir',
    type=str,
    default="/"+os.path.join("tmps", "visualize"),
    help='Where to save visualized sequences.'
)
  parser.add_argument(
    '--optical_flow_dir',
    type=str,
    default="/"+os.path.join("tmps", "cache", "optical_flow"),
    help='Where cached sequences are saved (otpical flow).'
)
  parser.add_argument(
    '--files_list',
    type=str,
    default="/"+os.path.join("tmps", "filestovisualize.txt"),
    help='List of files to visualize saved in, e.g., *.txt; if the list does not exist it will be generated (random 2 samples for each class).'
)
  parser.add_argument(
    "--clean", action="store_true",
                      help='Clean the visualize_dir (remove all exisiting files in the directory).'
)
  FLAGS, unparsed = parser.parse_known_args()
  output_dir = os.path.join(FLAGS.visualize_dir, os.path.split(FLAGS.optical_flow_dir)[1])
  if FLAGS.clean:
    prepare.remove_dir_tree(output_dir)
  prepare.ensure_dir_exists(output_dir)

  files = None
  if not os.path.exists(FLAGS.files_list):
    files = []
    for action in dataset.ACTION_LABELS:
      generator = glob(os.path.join(FLAGS.optical_flow_dir, "*", action+"*.npy"))
      for sample in random.sample(generator, 3):
        _, fn = os.path.split(sample)
        files.append(fn)
    with open(FLAGS.files_list, 'w+') as handler:
      for fn in files:
        handler.write(fn+"\n")
  else:
    with open(FLAGS.files_list, 'r') as handler:
      files = handler.read().split("\n")




