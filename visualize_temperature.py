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
from glob import glob

import numpy as np
import cv2

import imageio


FPS = 10

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--visualize_dir',
    type=str,
    default="/"+os.path.join("tmps", "visualize"),
    help='Where to save visualized sequences.'
)
  parser.add_argument(
    '--temperature_dir',
    type=str,
    default="/"+os.path.join("tmps", "cache", "temperature"),
    help='Where cached sequences are saved (temperature).'
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
  output_dir = os.path.join(FLAGS.visualize_dir, os.path.split(FLAGS.temperature_dir)[1])
  if FLAGS.clean:
    prepare.remove_dir_tree(output_dir)
  prepare.ensure_dir_exists(output_dir)

  files = None
  if not os.path.exists(FLAGS.files_list):
    files = []
    for action in dataset.ACTION_LABELS:
      generator = glob(os.path.join(FLAGS.flow_dir, "*", action+"*.npy"))
      for sample in random.sample(generator, 3):
        _, fn = os.path.split(sample)
        files.append(fn)
    with open(FLAGS.files_list, 'w+') as handler:
      for fn in files:
        handler.write(fn+"\n")
  else:
    with open(FLAGS.files_list, 'r') as handler:
      files = handler.read().split("\n")
  for name in files:
    if len(name) == 0:
      files.remove(name)

  temperatures = []
  for name in files:
    fn = glob(os.path.join(FLAGS.temperature_dir,"**",name))[0]
    temperature = np.load(fn)
    temperatures.append(temperature)

  grays = []
  for temperature in temperatures:
    gray = (255 * temperature).astype(np.uint8)
    grays.append(gray)

  def heatmap(sequence, cv_colormap: int = cv2.COLORMAP_JET):
    heatmap_flat = cv2.applyColorMap(
        sequence.flatten(), cv_colormap)
    return heatmap_flat.reshape(sequence.shape + (3,))

  bgrs = []
  for gray in grays:
    bgr = heatmap(gray)
    bgrs.append(bgr)
    
  for idx, bgr in enumerate(bgrs):
    fn = files[idx]
    gif_fn = fn.split(".")[0] + ".gif"
    with imageio.get_writer(os.path.join(output_dir, gif_fn), mode='I', duration=1/FPS) as writer:
      for frame in bgr:
        writer.append_data(frame[:, :, ::-1])



