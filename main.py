from tools import dataset
from tools.dataset import Dataset

import os
import numpy as np

dataset_dir = os.path.join("..", "dataset")
dataset_dir = os.path.join("tests", "data")
test_array = os.path.join("tests", "files", "array.csv")

if __name__ == "__main__":
    ds = Dataset(dataset_dir)