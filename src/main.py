from tools import dataset
from tools.dataset import Dataset

import os

dataset_dir = os.path.join("..", "..", "dataset")
dataset_dir = os.path.join("..", "tests", "datanosuchdirectory")

if __name__ == "__main__":
    ds = Dataset(dataset_dir)