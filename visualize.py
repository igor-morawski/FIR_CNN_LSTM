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

from tools import augmentation as augment


FPS = 10

def visualize_temperature(visualize_dir, temperature_dir, files_list, clean, augment_fct = None, suffix = ""):  
  output_dir = os.path.join(visualize_dir, os.path.split(temperature_dir)[1])
  if clean:
    prepare.remove_dir_tree(output_dir)
  prepare.ensure_dir_exists(output_dir)

  files = None
  if not os.path.exists(files_list):
    files = []
    for action in dataset.ACTION_LABELS:
      generator = glob(os.path.join(temperature_dir, "*", action+"*.npy"))
      for sample in random.sample(generator, 3):
        _, fn = os.path.split(sample)
        files.append(fn)
    with open(files_list, 'w+') as handler:
      for fn in files:
        handler.write(fn+"\n")
  else:
    with open(files_list, 'r') as handler:
      files = handler.read().split("\n")
  for name in files:
    if len(name) == 0:
      files.remove(name)

  temperatures = []
  for name in files:
    fn = glob(os.path.join(temperature_dir,"**",name))[0]
    temperature = np.load(fn)
    if augment_fct:
      temperature = augment_fct(temperature)
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
    gif_fn = fn.split(".")[0] + suffix + ".gif"
    with imageio.get_writer(os.path.join(output_dir, gif_fn), mode='I', duration=1/FPS) as writer:
      for frame in bgr:
        writer.append_data(frame[:, :, ::-1])

def visualize_flow(visualize_dir, flow_dir, files_list, clean, augment_fct = None, suffix = ""):
    
  output_dir = os.path.join(visualize_dir, os.path.split(flow_dir)[1])
  if clean:
    prepare.remove_dir_tree(output_dir)
  prepare.ensure_dir_exists(output_dir)

  files = None
  if not os.path.exists(files_list):
    files = []
    for action in dataset.ACTION_LABELS:
      generator = glob(os.path.join(flow_dir, "*", action+"*.npy"))
      for sample in random.sample(generator, 3):
        _, fn = os.path.split(sample)
        files.append(fn)
    with open(files_list, 'w+') as handler:
      for fn in files:
        handler.write(fn+"\n")
  else:
    with open(files_list, 'r') as handler:
      files = handler.read().split("\n")
  for name in files:
    if len(name) == 0:
      files.remove(name)
  
  flows = []
  for name in files:
    fn = glob(os.path.join(flow_dir,"**",name))[0]
    flow = np.load(fn)
    if augment_fct:
      flow = augment_fct(flow)
    flows.append(flow)

  def flow2bgr(flow_frame):
    hsv = np.zeros(flow_frame.shape[:-1] + (3,))
    mag, ang = cv2.cartToPolar(flow_frame[...,0], flow_frame[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

  bgrs = []
  for flow in flows:
    bgr = np.zeros(flow.shape[:-1] + (3,), dtype=np.uint8)
    for idx, frame in enumerate(flow):  
       bgr[idx] = flow2bgr(frame)
    bgrs.append(bgr)
  
  for idx, bgr in enumerate(bgrs):
    fn = files[idx]
    gif_fn = fn.split(".")[0] + suffix + ".gif"
    with imageio.get_writer(os.path.join(output_dir, gif_fn), mode='I', duration=1/FPS) as writer:
      for frame in bgr:
        writer.append_data(frame[:, :, ::-1])
      #write extra one frame to make up for the difference between optical flow and temperature sequences lengths
      writer.append_data(np.zeros_like(frame))


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
    '--flow_dir',
    type=str,
    default="/"+os.path.join("tmps", "cache", "optical_flow"),
    help='Where cached sequences are saved (optical flow).'
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
  parser.add_argument(
    "--augmentation", action="store_true",
                      help='Visualize augmentation.'
)

  FLAGS, unparsed = parser.parse_known_args()
  visualize_temperature(FLAGS.visualize_dir, FLAGS.temperature_dir, FLAGS.files_list, FLAGS.clean)
  visualize_flow(FLAGS.visualize_dir, FLAGS.flow_dir, FLAGS.files_list, FLAGS.clean)

  if FLAGS.augmentation:
    visualize_temperature(FLAGS.visualize_dir, FLAGS.temperature_dir, FLAGS.files_list, FLAGS.clean, augment_fct=augment.random_rotation, suffix="rot")
    visualize_flow(FLAGS.visualize_dir, FLAGS.flow_dir, FLAGS.files_list, FLAGS.clean, augment_fct=augment.random_rotation, suffix="rot")

    visualize_temperature(FLAGS.visualize_dir, FLAGS.temperature_dir, FLAGS.files_list, FLAGS.clean, augment_fct=augment.random_flip, suffix="flip")
    visualize_flow(FLAGS.visualize_dir, FLAGS.flow_dir, FLAGS.files_list, FLAGS.clean, augment_fct=augment.random_flip, suffix="flip")