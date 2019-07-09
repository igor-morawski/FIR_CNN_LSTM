from tools import dataset
from tools.dataset import Dataset
from tools import prepare

import os
import argparse

import numpy as np
import cv2

dataset_dir = os.path.join("..", "dataset")
data_notnormalized = Dataset(dataset_dir, minmax_normalized=False)
data_normalized = Dataset(dataset_dir, minmax_normalized=True)